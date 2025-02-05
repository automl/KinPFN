from kinpfn import bar_distribution, encoders, train, utils


def train_kinpfn(
    get_batch_function,
    seq_len,
    num_features,
    hps,
    epochs=20,
    buckets=1000,
    steps=500,
    batch_size=8,
    lr=0.001,
    emsize=128,
    nhead=4,
    nhid=256,
    nlayers=4,
    dropout=0.0,
    weight_decay=0.0,
    is_neps_run=False,
):

    print("Start Training")
    # define a bar distribution (riemann distribution) criterion with <buckets> bars
    print("Generate Data")
    ys = get_batch_function(100000, seq_len, num_features, hyperparameters=hps).target_y

    # we define our bar distribution adaptively with respect to the above sample of target ys from our prior
    bucket_limits = bar_distribution.get_bucket_limits(
        num_outputs=buckets, ys=ys, verbose=True
    )

    criterion = bar_distribution.FullSupportBarDistribution(bucket_limits)

    train_result = train.train(  # the prior is the key. It defines what we train on.
        priordataloader_class_or_get_batch=get_batch_function,
        criterion=criterion,
        # define the transformer size
        emsize=emsize,
        nhead=nhead,
        nhid=nhid,
        nlayers=nlayers,
        dropout=dropout,
        weight_decay=weight_decay,
        encoder_generator=encoders.Linear,
        y_encoder_generator=encoders.get_kinpfn_normalized_encoder(
            encoders.Linear, -6, 15
        ),
        # these are given to the prior, which needs to know how many features we have etc
        extra_prior_kwargs_dict={
            "num_features": num_features,
            "fuse_x_y": False,
            "hyperparameters": hps,
        },
        # change the number of epochs to put more compute into a training
        # an epoch length is defined by `steps_per_epoch`
        # the below means we do 10 epochs, with 100 batches per epoch and 4 datasets per batch
        # that means we look at 10*100*4 = 4000 datasets, typically we train on milllions of datasets.
        epochs=epochs,
        warmup_epochs=epochs // 4,
        steps_per_epoch=steps,
        batch_size=batch_size,
        # the lr is what you want to tune! usually something in [.00005,.0001,.0003,.001] works best
        # the lr interacts heavily with `batch_size` (smaller `batch_size` -> smaller best `lr`)
        lr=lr,
        # seq_len defines the size of your datasets (including the test set)
        seq_len=seq_len,
        # single_eval_pos_gen defines where to cut off between train and test set
        # a function that (randomly) returns lengths of the training set
        # the below definition, will just choose the size uniformly at random up to `seq_len`
        single_eval_pos_gen=utils.get_uniform_single_eval_pos_sampler(seq_len),
        progress_bar=True,
        is_neps_run=is_neps_run,
    )
    return train_result
