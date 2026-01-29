from train_model_rahma import proses_training

if __name__ == "__main__":
    print("SISTEM KLASIFIKASI SENTIMEN NAIVE BAYES")
    model, vectorizer = proses_training()
    print("Proses training dan evaluasi selesai.")
