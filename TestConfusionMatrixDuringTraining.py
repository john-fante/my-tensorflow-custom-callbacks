# Custom callback for creating a test set confusion matrix during training each 25 epochs

# test_dataset -> test tf.dataset
# target_names -> class names


class TestConfusionMatrixDuringTraining(Callback):
    def __init__(self, dataset, y_true, class_names, **kwargs):
        super(TestConfusionMatrixDuringTraining, self).__init__(**kwargs)
        self.dataset = dataset
        self.y_true = y_true
        self.class_names = class_names
        
    
    def on_epoch_end(self, epochs, logs = None):
        if (epochs + 1) % 25 == 0:
            test_take1 =  self.dataset.take(-1)
            test_take1_ = list(test_take1)
            pred = model.predict(test_take1)
            pred = np.round(pred).reshape(pred.shape[0])
            class_names_ = self.class_names
            
            cm = confusion_matrix(self.y_true , pred)
            cmd = ConfusionMatrixDisplay(cm, display_labels = list(class_names_.values()))

            fig, ax = plt.subplots(figsize=(3,3))
            ax.set_title('(Test Confusion Matrix) Epoch no :' + str(epochs + 1) +
                         '\n github.com/john-fante' + '\n kaggle.com/banddaniel' , color = 'red', fontsize = 10)
            cmd.plot(ax=ax,  cmap = 'Oranges', colorbar = False)
            plt.show()

     
        else:
            pass
