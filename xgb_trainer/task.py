import sys
import os
from datetime import datetime
import time

from argparse import ArgumentParser

import pandas as pd
from hypertune import HyperTune

from metadata import COLUMNS, FEATURES, TARGET, TRAIN_FILE, TRAIN_TEST_SPLIT, MODEL_FILE_NAME
import input
import model

import pickle


def main(args):

    print()
    print('Downloading train & test data...')
    TRAIN_PATH = '{}/{}'.format(args.data_dir, TRAIN_FILE)
    BUCKET_NAME, TRAIN_BLOB = input.split_gcs_path(TRAIN_PATH)
    input.download_blob(BUCKET_NAME, TRAIN_BLOB, TRAIN_FILE)

    print()
    print('Loading data...')
    with open(os.path.join('.', TRAIN_FILE), 'r') as train_data:
        train_data = pd.read_hdf(os.path.join('.', TRAIN_FILE))

    X = train_data.drop([TARGET], axis=1)
    y = train_data[TARGET]

    HYPERPARAMS = {
        'booster': 'gbtree',
        'verbosity': args.verbosity,
        'nthread': args.nthread,
        'eta': args.eta,
        'gamma': args.gamma,
        'max_depth': args.max_depth,
        'min_child_weight': args.min_child_weight,
        'max_delta_step': args.max_delta_step,
        'subsample': args.subsample,
        'colsample_bytree': args.colsample_bytree,
        'colsample_bylevel': args.colsample_bylevel,
        'colsample_bynode': args.colsample_bynode,
        'lambda': args.lmbda,
        'alpha': args.alpha,
        'tree_method': args.tree_method,
        'sketch_eps': args.sketch_eps,
        'scale_pos_weight': args.scale_pos_weight,
        'refresh_leaf': args.refresh_leaf,
        'process_type': args.process_type,
        'grow_policy': args.grow_policy,
        'max_leaves': args.max_leaves,
        'max_bin': args.max_bin,
        'predictor': args.predictor,
        'num_parallel_tree': args.num_parallel_tree,
        'objective': args.objective,
        'base_score': args.base_score,
        'eval_metric': args.eval_metric,
        'seed': args.seed
    }
    if args.eval_metric is None:
        HYPERPARAMS.pop('eval_metric')

    print()
    print('Training model...')
    print()
    print(HYPERPARAMS)
    bst = xgb.train(
        HYPERPARAMS,
        dtrain,
        num_boost_round=args.num_boost_rounds,
        early_stopping_rounds=args.early_stopping_rounds,
        evals=watchlist)

    print()
    print('Reporting accuracy score...')
    y_pred = bst.predict(dtest)
    predictions = [round(value) for value in y_pred]

    score = accuracy_score(y_test, predictions) * 100
    hpt = HyperTune()
    hpt.report_hyperparameter_tuning_metric(
        hyperparameter_metric_tag='accuracy',
        metric_value=score,
        global_step=1000)

    print()
    print('Accuracy score: {}'.format(score))

    print()
    print('Saving and exporting model to GCS location...')
    # Export the model to a file
    model_fn = '{}_{}{}'.format(MODEL_FILE_NAME, time.time(), '.pkl')
    with open(model_fn, 'wb') as f:
        pickle.dump(bst, f)

    DUMP_PATH = '{}/{}'.format(args.job_dir, model_fn)
    BUCKET_NAME, DUMP_BLOB = input.split_gcs_path(DUMP_PATH)
    input.upload_blob(BUCKET_NAME, DUMP_BLOB, model_fn)


if __name__ == "__main__":

    start = datetime.now()
    print()
    print('Training started at {}'.format(start))

    parser = ArgumentParser()
    parser.add_argument(
        '--data-dir',
        help='GCS or local paths to training data',
        required=True
    )
    parser.add_argument(
        '--job-dir',
        help='GCS location to write checkpoints and export models',
        required=True
    )
    parser.add_argument(
        '--verbosity',
        help='''\
                Verbosity of printing messages. Valid values are 0 (silent), 1 (warning), 2 (info), 3 (debug).
                Sometimes XGBoost tries to change configurations based on heuristics,
                which is displayed as warning message. If there’s unexpected behaviour,
                please try to increase value of verbosity.
            ''',
        default=1,
        choices=[0, 1, 2, 3]
    )
    parser.add_argument(
        '--nthread',
        help='''\
                Number of parallel threads used to run XGBoost.  Default to max if not specified
            ''',
        default=-1
    )
    parser.add_argument(
        '--num-boost-rounds',
        help='Number of boosting interations.',
        default=10,
        type=int
    )
    parser.add_argument(
        '--early-stopping-rounds',
        help='''\
                Activates early stopping. Validation error needs to decrease at least every (n) early_stopping_rounds round(s)
                to continue training. Requires at least one item in evals.
                If there’s more than one, will use the last. Returns the model from the last iteration (not the best one).
                If early stopping occurs, the model will have three additional fields: bst.best_score, bst.best_iteration and bst.best_ntree_limit.
                (Use bst.best_ntree_limit to get the correct value if num_parallel_tree and/or num_class appears in the parameters)
            ''',
        default=None,
        type=int
    )
    parser.add_argument(
        '--eta',
        help='''\
                Step size shrinkage used in update to prevents overfitting.
                After each boosting step, we can directly get the weights of new features,
                and eta shrinks the feature weights to make the boosting process more conservative.
                range: [0,1]
            ''',
        default=0.3,
        type=float
    )
    parser.add_argument(
        '--gamma',
        help='''\
                Minimum loss reduction required to make a further partition on a leaf node of the tree.
                The larger gamma is, the more conservative the algorithm will be.
                range: [0,∞]
            ''',
        default=0
    )
    parser.add_argument(
        '--max-depth',
        help='''\
                Maximum depth of a tree. Increasing this value will make the model more complex and more likely to overfit.
                0 indicates no limit. Note that limit is required when grow_policy is set of depthwise.
                range: [0,∞]
            ''',
        default=6
    )
    parser.add_argument(
        '--min-child-weight',
        help='''\
                Minimum sum of instance weight (hessian) needed in a child.
                If the tree partition step results in a leaf node with the sum of instance weight less than min_child_weight,
                then the building process will give up further partitioning. In linear regression task,
                this simply corresponds to minimum number of instances needed to be in each node.
                The larger min_child_weight is, the more conservative the algorithm will be.
                range: [0,∞]
            ''',
        default=1
    )
    parser.add_argument(
        '--max-delta-step',
        help='''\
                Maximum delta step we allow each leaf output to be.
                If the value is set to 0, it means there is no constraint.
                If it is set to a positive value, it can help making the update step more conservative.
                Usually this parameter is not needed, but it might help in logistic regression
                when class is extremely imbalanced. Set it to value of 1-10 might help control the update.
                range: [0,∞]
            ''',
        default=0
    )
    parser.add_argument(
        '--subsample',
        help='''\
                Subsample ratio of the training instances.
                Setting it to 0.5 means that XGBoost would randomly sample
                half of the training data prior to growing trees - and this will prevent overfitting.
                Subsampling will occur once in every boosting iteration.
                range: (0,1]
            ''',
        default=1,
        type=float
    )
    parser.add_argument(
        '--colsample-bytree',
        help='''\
                This is a family of parameters for subsampling of columns.
                All colsample_by* parameters have a range of (0, 1], the default value of 1,
                and colsample_bytree is the subsample ratio of columns when constructing each tree.
                Subsampling occurs once for every tree constructed.
                colsample_by* parameters work cumulatively.
                For instance, the combination {'colsample_bytree':0.5, 'colsample_bylevel':0.5, 'colsample_bynode':0.5}
                with 64 features will leave 4 features to choose from at each split
            ''',
        default=1,
        type=float
    )
    parser.add_argument(
        '--colsample-bylevel',
        help='''\
                This is a family of parameters for subsampling of columns.
                All colsample_by* parameters have a range of (0, 1], the default value of 1,
                and colsample_bylevel is the subsample ratio of columns for each level.
                Subsampling occurs once for every new depth level reached in a tree.
                Columns are subsampled from the set of columns chosen for the current tree.
                colsample_by* parameters work cumulatively.
                For instance, the combination {'colsample_bytree':0.5, 'colsample_bylevel':0.5, 'colsample_bynode':0.5}
                with 64 features will leave 4 features to choose from at each split
            ''',
        default=1,
        type=float
    )
    parser.add_argument(
        '--colsample-bynode',
        help='''\
                This is a family of parameters for subsampling of columns.
                All colsample_by* parameters have a range of (0, 1], the default value of 1,
                and colsample_bynode is the subsample ratio of columns for each node (split).
                Subsampling occurs once every time a new split is evaluated.
                Columns are subsampled from the set of columns chosen for the current level.
                colsample_by* parameters work cumulatively.
                For instance, the combination {'colsample_bytree':0.5, 'colsample_bylevel':0.5, 'colsample_bynode':0.5}
                with 64 features will leave 4 features to choose from at each split
            ''',
        default=1,
        type=float
    )
    parser.add_argument(
        '--lmbda',
        help='L2 regularization term on weights. Increasing this value will make model more conservative.',
        default=1)
    parser.add_argument(
        '--alpha',
        help='L1 regularization term on weights. Increasing this value will make model more conservative.',
        default=0)
    parser.add_argument(
        '--tree-method',
        help='''\
                The tree construction algorithm used in XGBoost. See description in the reference paper.
                Distributed and external memory version only support tree_method=approx.
                auto: Use heuristic to choose the fastest method.
                    -For small to medium dataset, exact greedy (exact) will be used.
                    -For very large dataset, approximate algorithm (approx) will be chosen.
                    -Because old behavior is always use exact greedy in single machine, user will get a message when approximate algorithm is chosen to notify this choice.
                exact: Exact greedy algorithm.
                approx: Approximate greedy algorithm using quantile sketch and gradient histogram.
                hist: Fast histogram optimized approximate greedy algorithm. It uses some performance improvements such as bins caching.
                gpu_exact: GPU implementation of exact algorithm.
                gpu_hist: GPU implementation of hist algorithm.
            ''',
        choices=['auto', 'exact', 'approx', 'hist', 'gpu_exact', 'gpu_hist'],
        default='auto',
    )
    parser.add_argument(
        '--sketch-eps',
        help='''\
                Only used for tree_method=approx.
                This roughly translates into O(1 / sketch_eps) number of bins.
                Compared to directly select number of bins, this comes with theoretical guarantee with sketch accuracy.
                Usually user does not have to tune this.
                But consider setting to a lower number for more accurate enumeration of split candidates.
                range: (0, 1)
            ''',
        default=0.03,
        type=float
    )
    parser.add_argument(
        '--scale-pos-weight',
        help='''\
                Control the balance of positive and negative weights, useful for unbalanced classes.
                A typical value to consider: sum(negative instances) / sum(positive instances)
            ''',
        default=1
    )
    parser.add_argument(
        '--refresh-leaf',
        help='''\
                This is a parameter of the refresh updater plugin. When this flag is 1,
                tree leafs as well as tree nodes’ stats are updated.
                When it is 0, only node stats are updated.
            ''',
        default=1
    )
    parser.add_argument(
        '--process-type',
        help='''\
                A type of boosting process to run.
                Choices: default, update
                    default: The normal boosting process which creates new trees.
                    update: Starts from an existing model and only updates its trees.
                    In each boosting iteration, a tree from the initial model is taken,
                    a specified sequence of updater plugins is run for that tree,
                    and a modified tree is added to the new model.
                    The new model would have either the same or smaller number of trees,
                    depending on the number of boosting iteratons performed.
                    Currently, the following built-in updater plugins could be meaningfully
                    used with this process type: refresh, prune. With process_type=update,
                    one cannot use updater plugins that create new trees.
            ''',
        choices=['default', 'update'],
        default='default'
    )
    parser.add_argument(
        '--grow-policy',
        help='''\
                Controls a way new nodes are added to the tree.
                Currently supported only if tree_method is set to hist.
                Choices: depthwise, lossguide
                    depthwise: split at nodes closest to the root.
                    lossguide: split at nodes with highest loss change.
            ''',
        choices=['depthwise', 'lossguide'],
        default='depthwise'
    )
    parser.add_argument(
        '--max-leaves',
        help='Maximum number of nodes to be added. Only relevant when grow_policy=lossguide is set.',
        default=0)
    parser.add_argument(
        '--max-bin',
        help='''\
                Only used if tree_method is set to hist
                Maximum number of discrete bins to bucket continuous features.
                Increasing this number improves the optimality of splits at the cost of higher computation time.
            ''',
        default=256
    )
    parser.add_argument(
        '--predictor',
        help='''\
            ''',
        choices=['cpu_predictor', 'gpu_predictor'],
        default='cpu_predictor'
    )
    parser.add_argument(
        '--num-parallel-tree',
        help='''\
                Number of parallel trees constructed during each iteration.
                This option is used to support boosted random forest.
            ''',
        default=1,
        type=int
    )
    parser.add_argument(
        '--objective',
        help='''\
                The learning objective.  For more details on learning tasks, visit XGBoost documentation @
                https://xgboost.readthedocs.io/en/latest/parameter.html
                ''',
        choices=[
            'reg:linear',
            'reg:logistic',
            'binary:logistic',
            'binary:logitraw',
            'binary:hinge',
            'count:poisson',
            'survival:cox',
            'multi:softmax',
            'multi:softprob',
            'rank:pairwise',
            'rank:ndcg',
            'rank:map',
            'reg:gamma',
            'reg:tweedie'
        ],
        default='binary:logistic'
    )
    parser.add_argument(
        '--base-score',
        help='''\
                The initial prediction score of all instances, global bias
                For sufficient number of iterations, changing this value will not have too much effect.
            ''',
        default=0.5,
        type=float
    )
    parser.add_argument(
        '--eval-metric',
        help='''\
                The evaluation metric.  For more details on learning tasks, visit XGBoost documentation @
                https://xgboost.readthedocs.io/en/latest/parameter.html
                If eval metric is not specified, XGBoost uses the appropriate eval metric based
                on the objective.
            '''
    )
    parser.add_argument(
        '--seed',
        help='Random number seed.',
        default=0
    )

    args = parser.parse_args()

    main(args)

    print()
    print('Training completed in {}'.format((datetime.now() - start)))
