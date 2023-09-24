
# Custom callback for creating a test set classification report during training

class TestClassificationReportDuringTraining(Callback):
  # test_dataset -> test tf.dataset
  # target_names -> class names
  # pred_round -> using for sigmoid output

    def on_epoch_end(self, epochs, logs = None ):
        test_take1 =  test_dataset.take(-1)
        test_take1_ = list(test_take1)
        pred = model.predict(test_take1)
        pred = np.round(pred).reshape(pred.shape[0])
        
        clf = classification_report(test_data['label'] , pred, target_names = list(classes.values()) )    
    
        print(Fore.RED + '~'*65)
        print(Style.RESET_ALL)
        print(Back.YELLOW + Fore.BLACK +'(Test Classification Report) Epoch no :' + str(epochs + 1))
        print(Style.RESET_ALL)
        print(Style.BRIGHT + Fore.BLACK + clf)
        print(Style.RESET_ALL)
        print(Fore.RED + '~'*65)
        print(Style.RESET_ALL)
