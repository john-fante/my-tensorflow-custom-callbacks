# Custom callback used saving models for each epoch

# for example -> '/kaggle/input/bird-sounds-h5/model_ResNet50_8_100_64.h5'
# model_name -> ResNet50
# epochs -> 8
# img_size -> 100
# batch_size -> 64

class ModelSaveEachEpoch(Callback):
    def __init__(self, model_name, img_size, batch_size, **kwargs):
        super(ModelSaveEachEpoch, self).__init__(**kwargs)
        self.model_name = model_name
        self.img_size = img_size
        self.batch_size = batch_size

    
    def on_epoch_end(self, epochs, logs = None ):
        file_name = 'model_' + self.model_name  + '_' + str(epochs + 1) + '_' + str(self.img_size) + '_' + str(self.batch_size) + '.h5'
        eval_ = model.evaluate(test_dataset)
        model.save(file_name)
        print(' ----- model saved ----- ')
