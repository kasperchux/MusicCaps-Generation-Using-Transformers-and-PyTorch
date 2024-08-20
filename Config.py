class Config(object):
    '''
    Just a class, what consider all
    hyperparameters of the model
    '''
    text_vocab_size = 10000
    music_vocab_size = 2048
    embedding_dim = 512
    d_model = 512
    n_head = 8
    num_encoder_layers = 6
    num_decoder_layers = 6
    ff_dim = 2048
    dropout = 0.1
    max_len = 512
    learning_rate = 1e-4
    epochs = 1
    batch_size = 64

config = Config() # Create an object to use it in other modules