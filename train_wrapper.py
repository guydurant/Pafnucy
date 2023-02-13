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


def load_csv(csv_file, data_dir):
    df = pd.read_csv(csv_file)
    protein_files = [os.path.join(data_dir, protein_file) for protein_file in df['protein']]
    ligand_files = [os.path.join(data_dir, ligand_file) for ligand_file in df['ligand']]
    keys = df['key']
    pks = df['pk']
    return protein_files, ligand_files, keys, pks

# def make_protein_pocket_file(protein_file, ligand_file, key):
#     cmd.load(protein_file, 'protein')
#     # cmd.load(f'temp_files/{protein_file.split("/")[-1].split(".")[0]}_charged.mol2', 'protein')
#     cmd.load(ligand_file, 'ligand')
#     cmd.select('pocket', 'byres (ligand around 6)')
#     cmd.save(f'temp_files/{key}_pocket.mol2', 'pocket')
#     cmd.delete('all')
#     return None

def vertices(Xc,Yc,Zc,SL):
    return [[Xc + SL/2, Yc + SL/2, Zc + SL/2],
            [Xc + SL/2, Yc + SL/2, Zc - SL/2],
            [Xc + SL/2, Yc - SL/2, Zc + SL/2],
            [Xc + SL/2, Yc - SL/2, Zc - SL/2],
            [Xc - SL/2, Yc + SL/2, Zc + SL/2],
            [Xc - SL/2, Yc + SL/2, Zc - SL/2],
            [Xc - SL/2, Yc - SL/2, Zc + SL/2],
            [Xc - SL/2, Yc - SL/2, Zc - SL/2]]

def make_20_box_around_ligand(protein_file, ligand_file, key):
    mol = Chem.MolFromMolFile(ligand_file)
    centroid = Chem.rdMolTransforms.ComputeCentroid(mol.GetConformer())
    box_vertices = vertices(centroid[0], centroid[1], centroid[2], 20)
    with open(protein_file, 'r') as f:
        lines = f.readlines()
    residues_in_pocket = []
    with open(f'temp_files/{key}_pocket.pdb', 'w') as f:
        for line in lines:
            if line.startswith('ATOM'):
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                if x < box_vertices[0][0] and x > box_vertices[4][0] and y < box_vertices[0][1] and y > box_vertices[2][1] and z < box_vertices[0][2] and z > box_vertices[1][2]:
                    residues_in_pocket.append(line[22:26])
        for line in lines:
            if line.startswith('ATOM') and line[22:26] in residues_in_pocket:
                f.write(line)
            
    return None

def calculate_charges_for_pocket(key):
    os.system(f'echo "open temp_files/{key}_pocket.pdb \n addh \n addcharge \n  save tmp.mol2 \n exit" | chimerax --nogui')
    # Do not use TIP3P atom types, pybel cannot read them
    os.system(f"sed 's/H\.t3p/H    /' tmp.mol2 | sed 's/O\.t3p/O\.3  /' > temp_files/{key}_pocket.mol2")



def get_pocket(protein_file, ligand_file, key):
    featurizer = Featurizer()
    make_20_box_around_ligand(protein_file, ligand_file, key)
    calculate_charges_for_pocket(key)
    pocket_file = f"temp_files/{key}_pocket.mol2"
    # try:
    pocket = next(pybel.readfile('mol2', pocket_file))
    # except:
    #     raise IOError('Cannot read %s file' % pocket_file)
    pocket_coords, pocket_features = featurizer.get_features(pocket, molcode=-1)
    os.system(f'rm temp_files/{key}_pocket.pdb')
    os.system(f'rm temp_files/{key}_pocket.mol2')
    return (pocket_coords, pocket_features)

def featurise_data(csv_file, data_dir):
    protein_files, ligand_files, keys, pks = load_csv(csv_file, data_dir)
    featurizer = Featurizer()
    with h5py.File(f"temp_features/{csv_file.split('/')[-1].split('.')[0]}.hdf", 'w') as f:
        for ligand_file in tqdm(ligand_files):
            # use filename without extension as dataset name
            name = keys[ligand_files.index(ligand_file)]

            # if args.verbose:
            #     print('reading %s' % ligand_file)
            try:
                ligand = next(pybel.readfile(ligand_file.split('.')[-1], ligand_file))
            except:
                raise IOError('Cannot read %s file' % ligand_file)

            ligand_coords, ligand_features = featurizer.get_features(ligand, molcode=1)
            pocket_coords, pocket_features = get_pocket(protein_files[ligand_files.index(ligand_file)], ligand_file, keys[ligand_files.index(ligand_file)])

            centroid = ligand_coords.mean(axis=0)
            ligand_coords -= centroid
            pocket_coords -= centroid

            data = np.concatenate(
                (np.concatenate((ligand_coords, pocket_coords)),
                np.concatenate((ligand_features, pocket_features))),
                axis=1,
            )

            dataset = f.create_dataset(name, data=data, shape=data.shape,
                                    dtype='float32', compression='lzf')
            dataset.attrs['affinity'] = pks[ligand_files.index(ligand_file)]

def get_batch(dataset_name, indices, rotation=0):
    featurizer = Featurizer()
    columns = {name: i for i, name in enumerate(featurizer.FEATURE_NAMES)}
    global coords, features, std
    x = []
    for i, idx in enumerate(indices):
        coords_idx = rotate(coords[dataset_name][idx], rotation)
        features_idx = features[dataset_name][idx]
        x.append(make_grid(coords_idx, features_idx,
                 grid_resolution=args.grid_spacing,
                 max_dist=args.max_dist))
    x = np.vstack(x)
    x[..., columns['partialcharge']] /= std
    return x

def train_model(args):
    ids = {}
    affinity = {}
    coords = {}
    features = {}
    datasets = {'training': args.csv_file, 'valididation': args.valid_csv_file}

    featurizer = Featurizer()
    columns = {name: i for i, name in enumerate(featurizer.FEATURE_NAMES)}

    for dictionary in [ids, affinity, coords, features]:
        for dataset_name in datasets.keys():
            dictionary[dataset_name] = []

    for dataset_name in datasets.keys():
        dataset_path = f'temp_features/{datasets[dataset_name].split("/")[-1].split(".")[0]}_features.pkl'
        with h5py.File(dataset_path, 'r') as f:
            for pdb_id in f:
                dataset = f[pdb_id]

                coords[dataset_name].append(dataset[:, :3])
                features[dataset_name].append(dataset[:, 3:])
                affinity[dataset_name].append(dataset.attrs['affinity'])
                ids[dataset_name].append(pdb_id)

        ids[dataset_name] = np.array(ids[dataset_name])
        affinity[dataset_name] = np.reshape(affinity[dataset_name], (-1, 1))


    # normalize charges
    charges = []
    for feature_data in features['training']:
        charges.append(feature_data[..., columns['partialcharge']])

    charges = np.concatenate([c.flatten() for c in charges])

    m = charges.mean()
    std = charges.std()
    print('charges: mean=%s, sd=%s' % (m, std))
    print('use sd as scaling factor')




    print('\n---- DATA ----\n')

    tmp = get_batch('training', range(min(50, len(features['training']))))

    assert ((tmp[:, :, :, :, columns['molcode']] == 0.0).any()
            and (tmp[:, :, :, :, columns['molcode']] == 1.0).any()
            and (tmp[:, :, :, :, columns['molcode']] == -1.0).any()).all()

    idx1 = [[i[0]] for i in np.where(tmp[:, :, :, :, columns['molcode']] == 1.0)]
    idx2 = [[i[0]] for i in np.where(tmp[:, :, :, :, columns['molcode']] == -1.0)]

    # print('\nexamples:')
    # for mtype, mol in [['ligand', tmp[idx1]], ['protein', tmp[idx2]]]:
    #     print(' ', mtype)
    #     for name, num in columns.items():
    #         print('  ', name, mol[0, num])
    #     print('')


    # Best error we can get without any training (MSE from training set mean):
    t_baseline = ((affinity['training'] - affinity['training'].mean()) ** 2.0).mean()
    v_baseline = ((affinity['validation'] - affinity['training'].mean()) ** 2.0).mean()
    # print('baseline mse: training=%s, validation=%s' % (t_baseline, v_baseline))


    # NET PARAMS

    ds_sizes = {dataset: len(affinity[dataset]) for dataset in datasets.keys()}
    _, isize, *_, in_chnls = get_batch('training', [0]).shape
    osize = 1

    for set_name, set_size in ds_sizes.items():
        print('%s %s samples' % (set_size, set_name))

    num_batches = {dataset: ceil(size / args.batch_size)
                for dataset, size in ds_sizes.items()}
    graph = net.make_SB_network(isize=isize, in_chnls=in_chnls, osize=osize,
                                  conv_patch=args.conv_patch,
                                  pool_patch=args.pool_patch,
                                  conv_channels=args.conv_channels,
                                  dense_sizes=args.dense_sizes,
                                  lmbda=args.lmbda,
                                  learning_rate=args.learning_rate)


    # train_writer = tf.summary.FileWriter(os.path.join(logdir, 'training_set'),
    #                                     graph, flush_secs=1)
    # val_writer = tf.summary.FileWriter(os.path.join(logdir, 'validation_set'),
    #                                 flush_secs=1)

    net_summaries, training_summaries = net.make_summaries_SB(graph)

    x = graph.get_tensor_by_name('input/structure:0')
    y = graph.get_tensor_by_name('output/prediction:0')
    t = graph.get_tensor_by_name('input/affinity:0')
    keep_prob = graph.get_tensor_by_name('fully_connected/keep_prob:0')
    train = graph.get_tensor_by_name('training/train:0')
    mse = graph.get_tensor_by_name('training/mse:0')
    feature_importance = graph.get_tensor_by_name('net_properties/feature_importance:0')
    global_step = graph.get_tensor_by_name('training/global_step:0')

    convs = '_'.join((str(i) for i in args.conv_channels))
    fcs = '_'.join((str(i) for i in args.dense_sizes))

    with graph.as_default():
        saver = tf.train.Saver(max_to_keep=args.to_keep)
    
    def batches(set_name):
        """Batch generator, yields slice indices"""
        global num_batches, args, ds_sizes
        for b in range(num_batches[set_name]):
            bi = b * args.batch_size
            bj = (b + 1) * args.batch_size
            if b == num_batches[set_name] - 1:
                bj = ds_sizes[set_name]
            yield bi, bj
    
    err = float('inf')

    train_sample = min(args.batch_size, len(features['training']))
    val_sample = min(args.batch_size, len(features['validation']))

    print('\n---- TRAINING ----\n')
    with tf.Session(graph=graph) as session:
        session.run(tf.global_variables_initializer())

        summary_imp = tf.Summary()
        feature_imp = session.run(feature_importance)
        image = net.feature_importance_plot(feature_imp)
        summary_imp.value.add(tag='feature_importance_%s' % 0, image=image)
        # train_writer.add_summary(summary_imp, 0)

        stats_net = session.run(
            net_summaries,
            feed_dict={x: get_batch('training', range(train_sample)),
                    t: affinity['training'][:train_sample],
                    keep_prob: 1.0}
        )

        # train_writer.add_summary(stats_net, 0)

        for epoch in range(args.num_epochs):
            for rotation in args.rotations:
                print('rotation', rotation)
                # TRAIN #
                x_t, y_t = shuffle(range(ds_sizes['training']), affinity['training'])

                for bi, bj in batches('training'):
                    session.run(train, feed_dict={x: get_batch('training',
                                                            x_t[bi:bj],
                                                            rotation),
                                                t: y_t[bi:bj], keep_prob: args.kp})

                # SAVE STATS - per rotation #
                stats_t, stats_net = session.run(
                    [training_summaries, net_summaries],
                    feed_dict={x: get_batch('training', x_t[:train_sample]),
                            t: y_t[:train_sample],
                            keep_prob: 1.0}
                )

                # train_writer.add_summary(stats_t, global_step.eval())
                # train_writer.add_summary(stats_net, global_step.eval())

                stats_v = session.run(
                    training_summaries,
                    feed_dict={x: get_batch('validation', range(val_sample)),
                            t: affinity['validation'][:val_sample],
                            keep_prob: 1.0}
                )

                # val_writer.add_summary(stats_v, global_step.eval())

            # SAVE STATS - per epoch #
            # training set error
            pred_t = np.zeros((ds_sizes['training'], 1))
            mse_t = np.zeros(num_batches['training'])

            for b, (bi, bj) in enumerate(batches('training')):
                weight = (bj - bi) / ds_sizes['training']

                pred_t[bi:bj], mse_t[b] = session.run(
                    [y, mse],
                    feed_dict={x: get_batch('training', x_t[bi:bj]),
                            t: y_t[bi:bj],
                            keep_prob: 1.0}
                )

                mse_t[b] *= weight

            mse_t = mse_t.sum()

            summary_mse = tf.Summary()
            summary_mse.value.add(tag='mse_all', simple_value=mse_t)
            # train_writer.add_summary(summary_mse, global_step.eval())

            # predictions distribution
            summary_pred = tf.Summary()
            summary_pred.value.add(tag='predictions_all',
                                histo=net.custom_summary_histogram(pred_t))
            # train_writer.add_summary(summary_pred, global_step.eval())

            # validation set error
            mse_v = 0
            for bi, bj in batches('validation'):
                weight = (bj - bi) / ds_sizes['validation']
                mse_v += weight * session.run(
                    mse,
                    feed_dict={x: get_batch('validation', range(bi, bj)),
                            t: affinity['validation'][bi:bj],
                            keep_prob: 1.0}
                )

            summary_mse = tf.Summary()
            summary_mse.value.add(tag='mse_all', simple_value=mse_v)
            # val_writer.add_summary(summary_mse, global_step.eval())

            # SAVE MODEL #
            print('epoch: %s train error: %s, validation error: %s'
                % (epoch, mse_t, mse_v))

            if mse_v <= err:
                err = mse_v
                checkpoint = saver.save(session, args.model_name, global_step=global_step)

                # feature importance
                summary_imp = tf.Summary()
                feature_imp = session.run(feature_importance)
                image = net.feature_importance_plot(feature_imp)
                summary_imp.value.add(tag='feature_importance', image=image)
                # train_writer.add_summary(summary_imp, global_step.eval())


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_file', type=str, default='train.csv')
    parser.add_argument('--val_csv_file', type=str, default='val.csv')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--val_data_dir', type=str, default='data')
    parser.add_argument('--model_name', type=str, default='test')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--predict', action='store_true')

    args = parser.parse_args()
    if args.train:
        if not os.path.exists(f'temp_features/{args.csv_file.split("/")[-1].split(".")[0]}_features.pkl'):
            print('Extracting features...')
            featurise_data(args.csv_file, args.data_dir)
        if not os.path.exists(f'temp_features/{args.val_csv_file.split("/")[-1].split(".")[0]}_features.pkl'):
            print('Extracting features...')
            featurise_data(args.val_csv_file, args.val_data_dir)
        train_model(args)
    elif args.predict:
        if not os.path.exists(f'temp_features/{args.val_csv_file.split("/")[-1].split(".")[0]}_features.pkl'):
            print('Extracting features...')
            featurise_data(args.val_csv_file, args.val_data_dir)
    else:
        raise ValueError('No action specified')
        
        


