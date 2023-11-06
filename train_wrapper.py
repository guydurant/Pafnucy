import os
import pandas as pd
from featuriser import Featurizer, make_grid, rotate
import net
from openbabel import pybel

# from pymol import cmd
import numpy as np
import h5py
import tensorflow as tf
from rdkit import Chem
from rdkit.Chem import rdMolTransforms
from math import sqrt, ceil
from sklearn.utils import shuffle
from tqdm import tqdm
from glob import glob


def load_csv(csv_file, data_dir):
    df = pd.read_csv(csv_file)
    protein_files = [
        os.path.join(data_dir, protein_file) for protein_file in df["protein"]
    ]
    ligand_files = [os.path.join(data_dir, ligand_file) for ligand_file in df["ligand"]]
    keys = df["key"]
    pks = df["pk"]
    return protein_files, ligand_files, keys, pks


def vertices(Xc, Yc, Zc, SL):
    return [
        [Xc + SL / 2, Yc + SL / 2, Zc + SL / 2],
        [Xc + SL / 2, Yc + SL / 2, Zc - SL / 2],
        [Xc + SL / 2, Yc - SL / 2, Zc + SL / 2],
        [Xc + SL / 2, Yc - SL / 2, Zc - SL / 2],
        [Xc - SL / 2, Yc + SL / 2, Zc + SL / 2],
        [Xc - SL / 2, Yc + SL / 2, Zc - SL / 2],
        [Xc - SL / 2, Yc - SL / 2, Zc + SL / 2],
        [Xc - SL / 2, Yc - SL / 2, Zc - SL / 2],
    ]


def read_pdb_line(line):
    atom_type = line[0:6].strip()
    atom_number = line[6:11].strip()
    atom_name = line[12:16].strip()
    alt_loc = line[16:17].strip()
    res_name = line[17:20].strip()
    chain_id = line[21:22].strip()
    res_number = line[22:26].strip()
    icode = line[26:27].strip()
    x = float(line[30:38].strip())
    y = float(line[38:46].strip())
    z = float(line[46:54].strip())
    occupancy = float(line[54:60].strip())
    temp_factor = float(line[60:66].strip())
    element = line[76:78].strip()
    charge = line[78:80].strip()
    return x, y, z, res_number, chain_id


def make_20_box_around_ligand(protein_file, ligand_file, key, csv_file):
    if not os.path.exists(f'data/scratch/{csv_file.split("/")[-1].split(".")[0]}'):
        os.makedirs(f'data/scratch/{csv_file.split("/")[-1].split(".")[0]}')
    mol = Chem.MolFromMolFile(ligand_file)
    centroid = Chem.rdMolTransforms.ComputeCentroid(mol.GetConformer())
    box_vertices = vertices(centroid[0], centroid[1], centroid[2], 20)
    with open(protein_file, "r") as f:
        lines = f.readlines()
    residues_in_pocket = []
    with open(
        f'data/scratch/{csv_file.split("/")[-1].split(".")[0]}/{key}_pocket.pdb', "w"
    ) as f:
        for line in lines:
            if line.startswith("ATOM"):
                x, y, z, res_number, chain_id = read_pdb_line(line)
                if (
                    x < box_vertices[0][0]
                    and x > box_vertices[4][0]
                    and y < box_vertices[0][1]
                    and y > box_vertices[2][1]
                    and z < box_vertices[0][2]
                    and z > box_vertices[1][2]
                ):
                    residues_in_pocket.append((res_number, chain_id))
        for line in lines:
            if line.startswith("ATOM"):
                x, y, z, res_number, chain_id = read_pdb_line(line)
                if (res_number, chain_id) in residues_in_pocket:
                    f.write(line)

    return None


def get_pocket(protein_file, ligand_file, key, csv_file):
    featurizer = Featurizer()
    make_20_box_around_ligand(protein_file, ligand_file, key, csv_file)
    pocket_file = (
        f"data/scratch/{csv_file.split('/')[-1].split('.')[0]}/{key}_pocket.pdb"
    )
    pocket = next(pybel.readfile("pdb", pocket_file))
    pocket_coords, pocket_features = featurizer.get_features(pocket, molcode=-1)
    os.system(
        f"rm data/scratch/{csv_file.split('/')[-1].split('.')[0]}/{key}_pocket.pdb"
    )
    return (pocket_coords, pocket_features)


def featurise_data(csv_file, data_dir):
    protein_files, ligand_files, keys, pks = load_csv(csv_file, data_dir)
    featurizer = Featurizer()
    if not os.path.exists("data/features"):
        os.makedirs("data/features")
    with h5py.File(
        f"data/features/{csv_file.split('/')[-1].split('.')[0]}_features.hdf", "w"
    ) as f:
        for ligand_file in tqdm(ligand_files):
            # use filename without extension as dataset name
            name = keys[ligand_files.index(ligand_file)]

            # if args.verbose:
            #     print('reading %s' % ligand_file)
            try:
                ligand = next(pybel.readfile(ligand_file.split(".")[-1], ligand_file))
            except:
                raise IOError("Cannot read %s file" % ligand_file)

            ligand_coords, ligand_features = featurizer.get_features(ligand, molcode=1)
            pocket_coords, pocket_features = get_pocket(
                protein_files[ligand_files.index(ligand_file)],
                ligand_file,
                keys[ligand_files.index(ligand_file)],
                csv_file,
            )

            centroid = ligand_coords.mean(axis=0)
            ligand_coords -= centroid
            pocket_coords -= centroid

            data = np.concatenate(
                (
                    np.concatenate((ligand_coords, pocket_coords)),
                    np.concatenate((ligand_features, pocket_features)),
                ),
                axis=1,
            )

            dataset = f.create_dataset(
                name, data=data, shape=data.shape, dtype="float32", compression="lzf"
            )
            dataset.attrs["affinity"] = pks[ligand_files.index(ligand_file)]


def get_batch(
    dataset_name,
    indices,
    coords,
    features,
    std,
    rotation=0,
):
    featurizer = Featurizer()
    columns = {name: i for i, name in enumerate(featurizer.FEATURE_NAMES)}
    # global coords, features, std
    x = []
    for i, idx in enumerate(indices):
        coords_idx = rotate(coords[dataset_name][idx], rotation)
        features_idx = features[dataset_name][idx]
        x.append(
            make_grid(
                coords_idx,
                features_idx,
                grid_resolution=args.grid_spacing,
                max_dist=args.max_dist,
            )
        )
    x = np.vstack(x)
    x[..., columns["partialcharge"]] /= std
    return x


def train_model(args):
    ids = {}
    affinity = {}
    coords = {}
    features = {}
    datasets = {
        "training": args.csv_file,
        "validation": None,
    }  # could be set as args.val_csv_file

    featurizer = Featurizer()
    columns = {name: i for i, name in enumerate(featurizer.FEATURE_NAMES)}

    for dictionary in [ids, affinity, coords, features]:
        for dataset_name in datasets.keys():
            dictionary[dataset_name] = []

    for dataset_name in datasets.keys():
        if datasets[dataset_name] is not None:
            dataset_path = f'data/features/{datasets[dataset_name].split("/")[-1].split(".")[0]}_features.hdf'
            with h5py.File(dataset_path, "r") as f:
                for pdb_id in f:
                    dataset = f[pdb_id]

                    coords[dataset_name].append(dataset[:, :3])
                    features[dataset_name].append(dataset[:, 3:])
                    affinity[dataset_name].append(dataset.attrs["affinity"])
                    ids[dataset_name].append(pdb_id)

            ids[dataset_name] = np.array(ids[dataset_name])
            affinity[dataset_name] = np.reshape(affinity[dataset_name], (-1, 1))
        else:
            # take sample from training set
            indices = np.random.choice(
                len(ids["training"]), size=1000, replace=False
            ).tolist()
            ids[dataset_name] = [ids["training"][i] for i in indices]
            ids["training"] = np.delete(ids["training"], indices, axis=0)
            affinity[dataset_name] = [affinity["training"][i] for i in indices]
            affinity["training"] = np.delete(affinity["training"], indices, axis=0)
            coords[dataset_name] = [coords["training"][i] for i in indices]
            coords["training"] = np.delete(coords["training"], indices, axis=0)
            features[dataset_name] = [features["training"][i] for i in indices]
            features["training"] = np.delete(features["training"], indices, axis=0)
    # normalize charges
    charges = []
    for feature_data in features["training"]:
        charges.append(feature_data[..., columns["partialcharge"]])

    charges = np.concatenate([c.flatten() for c in charges])

    m = charges.mean()
    std = charges.std()
    print("charges: mean=%s, sd=%s" % (m, std))
    print("use sd as scaling factor")

    print("\n---- DATA ----\n")

    tmp = get_batch(
        "training", range(min(50, len(features["training"]))), coords, features, std
    )

    assert (
        (tmp[:, :, :, :, columns["molcode"]] == 0.0).any()
        and (tmp[:, :, :, :, columns["molcode"]] == 1.0).any()
        and (tmp[:, :, :, :, columns["molcode"]] == -1.0).any()
    ).all()

    idx1 = [[i[0]] for i in np.where(tmp[:, :, :, :, columns["molcode"]] == 1.0)]
    idx2 = [[i[0]] for i in np.where(tmp[:, :, :, :, columns["molcode"]] == -1.0)]

    # print('\nexamples:')
    # for mtype, mol in [['ligand', tmp[idx1]], ['protein', tmp[idx2]]]:
    #     print(' ', mtype)
    #     for name, num in columns.items():
    #         print('  ', name, mol[0, num])
    #     print('')

    # Best error we can get without any training (MSE from training set mean):
    t_baseline = ((affinity["training"] - affinity["training"].mean()) ** 2.0).mean()
    v_baseline = ((affinity["validation"] - affinity["training"].mean()) ** 2.0).mean()
    # print('baseline mse: training=%s, validation=%s' % (t_baseline, v_baseline))

    # NET PARAMS

    ds_sizes = {dataset: len(affinity[dataset]) for dataset in datasets.keys()}
    _, isize, *_, in_chnls = get_batch("training", [0], coords, features, std).shape
    osize = 1

    for set_name, set_size in ds_sizes.items():
        print("%s %s samples" % (set_size, set_name))

    num_batches = {
        dataset: ceil(size / args.batch_size) for dataset, size in ds_sizes.items()
    }
    graph = net.make_SB_network(
        isize=isize,
        in_chnls=in_chnls,
        osize=osize,
        conv_patch=args.conv_patch,
        pool_patch=args.pool_patch,
        conv_channels=args.conv_channels,
        dense_sizes=args.dense_sizes,
        lmbda=args.lmbda,
        learning_rate=args.learning_rate,
    )

    # train_writer = tf.summary.FileWriter(os.path.join(logdir, 'training_set'),
    #                                     graph, flush_secs=1)
    # val_writer = tf.summary.FileWriter(os.path.join(logdir, 'validation_set'),
    #                                 flush_secs=1)

    net_summaries, training_summaries = net.make_summaries_SB(graph)

    x = graph.get_tensor_by_name("input/structure:0")
    y = graph.get_tensor_by_name("output/prediction:0")
    t = graph.get_tensor_by_name("input/affinity:0")
    keep_prob = graph.get_tensor_by_name("fully_connected/keep_prob:0")
    train = graph.get_tensor_by_name("training/train:0")
    mse = graph.get_tensor_by_name("training/mse:0")
    feature_importance = graph.get_tensor_by_name("net_properties/feature_importance:0")
    global_step = graph.get_tensor_by_name("training/global_step:0")

    convs = "_".join((str(i) for i in args.conv_channels))
    fcs = "_".join((str(i) for i in args.dense_sizes))

    with graph.as_default():
        saver = tf.train.Saver(max_to_keep=args.to_keep)

    def batches(set_name):
        """Batch generator, yields slice indices"""
        for b in range(num_batches[set_name]):
            bi = b * args.batch_size
            bj = (b + 1) * args.batch_size
            if b == num_batches[set_name] - 1:
                bj = ds_sizes[set_name]
            yield bi, bj

    err = float("inf")

    train_sample = min(args.batch_size, len(features["training"]))
    val_sample = min(args.batch_size, len(features["validation"]))

    print("\n---- TRAINING ----\n")
    if not os.path.exists("data/models"):
        os.makedirs("data/models")
    with tf.Session(graph=graph) as session:
        session.run(tf.global_variables_initializer())

        summary_imp = tf.Summary()
        feature_imp = session.run(feature_importance)
        image = net.feature_importance_plot(feature_imp)
        summary_imp.value.add(tag="feature_importance_%s" % 0, image=image)

        for epoch in tqdm(range(args.num_epochs)):
            for rotation in args.rotations:
                # TRAIN #
                x_t, y_t = shuffle(range(ds_sizes["training"]), affinity["training"])

                for bi, bj in batches("training"):
                    session.run(
                        train,
                        feed_dict={
                            x: get_batch(
                                "training", x_t[bi:bj], coords, features, std, rotation
                            ),
                            t: y_t[bi:bj],
                            keep_prob: args.kp,
                        },
                    )

            pred_t = np.zeros((ds_sizes["training"], 1))
            mse_t = np.zeros(num_batches["training"])

            for b, (bi, bj) in enumerate(batches("training")):
                weight = (bj - bi) / ds_sizes["training"]

                pred_t[bi:bj], mse_t[b] = session.run(
                    [y, mse],
                    feed_dict={
                        x: get_batch("training", x_t[bi:bj], coords, features, std),
                        t: y_t[bi:bj],
                        keep_prob: 1.0,
                    },
                )

                mse_t[b] *= weight

            mse_t = mse_t.sum()

            summary_mse = tf.Summary()
            summary_mse.value.add(tag="mse_all", simple_value=mse_t)
            summary_pred = tf.Summary()
            summary_pred.value.add(
                tag="predictions_all", histo=net.custom_summary_histogram(pred_t)
            )
            mse_v = 0
            for bi, bj in batches("validation"):
                weight = (bj - bi) / ds_sizes["validation"]
                mse_v += weight * session.run(
                    mse,
                    feed_dict={
                        x: get_batch(
                            "validation", range(bi, bj), coords, features, std
                        ),
                        t: affinity["validation"][bi:bj],
                        keep_prob: 1.0,
                    },
                )

            summary_mse = tf.Summary()
            summary_mse.value.add(tag="mse_all", simple_value=mse_v)
            # SAVE MODEL #
            print(
                "epoch: %s train error: %s, validation error: %s"
                % (epoch, mse_t, mse_v)
            )

            if mse_v <= err:
                err = mse_v
                checkpoint = saver.save(
                    session, "data/models/" + args.model_name, global_step=global_step
                )

                # feature importance
                summary_imp = tf.Summary()
                feature_imp = session.run(feature_importance)
                image = net.feature_importance_plot(feature_imp)
                summary_imp.value.add(tag="feature_importance", image=image)


def __get_batch(args, charge_column, coords, features):
    batch_grid = []

    for crd, f in zip(coords, features):
        batch_grid.append(
            make_grid(crd, f, max_dist=args.max_dist, grid_resolution=args.grid_spacing)
        )
        if len(batch_grid) == args.batch_size:
            # if batch is not specified it will never happen
            batch_grid = np.vstack(batch_grid)
            batch_grid[..., charge_column] /= args.charge_scaler
            yield batch_grid
            batch_grid = []

    if len(batch_grid) > 0:
        batch_grid = np.vstack(batch_grid)
        batch_grid[..., charge_column] /= args.charge_scaler
        yield batch_grid


def predict(args):
    featurizer = Featurizer()

    charge_column = featurizer.FEATURE_NAMES.index("partialcharge")

    coords = []
    features = []
    names = []
    pks = []

    input_file = (
        f'data/features/{args.val_csv_file.split("/")[-1].split(".")[0]}_features.hdf'
    )

    with h5py.File(input_file, "r") as f:
        for name in f:
            names.append(name)
            dataset = f[name]
            coords.append(dataset[:, :3])
            features.append(dataset[:, 3:])
            pks.append(dataset.attrs["affinity"])

    meta_name = glob(f"data/models/{args.model_name}-*.meta")
    # get the latest model
    meta_name_latest = max(meta_name, key=os.path.getctime)

    saver = tf.train.import_meta_graph(meta_name_latest, clear_devices=True)

    predict = tf.get_collection("output")[0]
    inp = tf.get_collection("input")[0]
    kp = tf.get_collection("kp")[0]

    # if args.verbose:
    #     print('restored network from %s\n' % args.network)

    with tf.Session() as session:
        saver.restore(session, meta_name_latest.split(".meta")[0])
        predictions = []
        batch_generator = __get_batch(args, charge_column, coords, features)
        for grid in batch_generator:
            # TODO: remove kp in next release
            # it's here for backward compatibility
            predictions.append(session.run(predict, feed_dict={inp: grid, kp: 1.0}))
    return np.concatenate(predictions).reshape(-1), names, pks


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", type=str, default="train.csv")
    parser.add_argument("--val_csv_file", type=str)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--val_data_dir", type=str)
    parser.add_argument("--model_name", type=str, default="test")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--cache", action="store_true")
    parser.add_argument(
        "--grid_spacing",
        "-g",
        default=1.0,
        type=float,
        help="distance between grid points",
    )
    parser.add_argument(
        "--max_dist",
        "-d",
        default=10.0,
        type=float,
        help="max distance from complex center",
    )
    parser.add_argument(
        "--conv_patch", default=5, type=int, help="patch size for convolutional layers"
    )
    parser.add_argument(
        "--pool_patch", default=2, type=int, help="patch size for pooling layers"
    )
    parser.add_argument(
        "--conv_channels",
        metavar="C",
        default=[64, 128, 256],
        type=int,
        nargs="+",
        help="number of fileters in convolutional layers",
    )
    parser.add_argument(
        "--dense_sizes",
        metavar="D",
        default=[1000, 500, 200],
        type=int,
        nargs="+",
        help="number of neurons in dense layers",
    )
    parser.add_argument(
        "--keep_prob",
        dest="kp",
        default=0.5,
        type=float,
        help="keep probability for dropout",
    )
    parser.add_argument(
        "--l2", dest="lmbda", default=0.001, type=float, help="lambda for weight decay"
    )
    parser.add_argument(
        "--rotations",
        metavar="R",
        default=list(range(24)),
        type=int,
        nargs="+",
        help="rotations to perform",
    )
    parser.add_argument(
        "--learning_rate", default=1e-5, type=float, help="learning rate"
    )
    parser.add_argument("--batch_size", default=20, type=int, help="batch size")
    parser.add_argument("--num_epochs", default=20, type=int, help="number of epochs")
    parser.add_argument(
        "--num_checkpoints",
        dest="to_keep",
        default=10,
        type=int,
        help="number of checkpoints to keep",
    )
    parser.add_argument(
        "--charge_scaler",
        type=float,
        default=0.425896,
        help="scaling factor for the charge"
        " (use the same factor when preparing data for"
        " training and and for predictions)",
    )

    args = parser.parse_args()
    if args.train:
        if not os.path.exists(
            f'data/features/{args.csv_file.split("/")[-1].split(".")[0]}_features.hdf'
        ):
            featurise_data(args.csv_file, args.data_dir)
        print("Training model...")
        train_model(args)
    if args.predict:
        if not os.path.exists(
            f'data/features/{args.val_csv_file.split("/")[-1].split(".")[0]}_features.hdf'
        ):
            print("Extracting features...")
            featurise_data(args.val_csv_file, args.val_data_dir)
        proteins, ligands, keys, pks = load_csv(args.val_csv_file, args.val_data_dir)
        predictions, reordered_keys, reordered_pks = predict(args)
        df = pd.DataFrame(
            {"key": reordered_keys, "pred": predictions, "pk": reordered_pks}
        )
        if not os.path.exists("data/results"):
            os.makedirs("data/results")
        df.to_csv(
            f'data/results/{args.model_name}_{args.val_csv_file.split("/")[-1]}',
            index=False,
        )
    else:
        raise ValueError("Please specify --train or --predict or both.")
