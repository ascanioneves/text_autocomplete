import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import tensorflow as tf
import keras_tuner as kt
import matplotlib.pyplot as plt # type: ignore

df = pd.read_csv('train.csv', header=None, names=['ClassIndex', 'Título', 'Descrição'])

df['text'] = df['Título'] + ' ' + df['Descrição']

df['ClassIndex'] = df['ClassIndex'] - 1

VOCAB_SIZE = 1000
X_train, X_test, y_train,y_test = train_test_split(df['text'].values, df['ClassIndex'].values, test_size=0.2, random_state=4266)

encoder = tf.keras.layers.TextVectorization(max_tokens=VOCAB_SIZE)
encoder.adapt(X_train)

#Hp = hiperparâmetros
def build_model(hp):
    model = tf.keras.Sequential([
        encoder,
        tf.keras.layers.Embedding(
            input_dim=len(encoder.get_vocabulary()),
            output_dim=hp.Int('embedding_dim', min_value=32, max_value=128, step=32),
            mask_zero=True
        ),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
            units=hp.Int('lstm_units', min_value=32, max_value=128, step=32),
            return_sequences=True    
        )), 
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
            units=hp.Int('lstm_units', min_value=16, max_value=64, step=16)   
        )), 
        tf.keras.layers.Dense(
            units=hp.Int('dense_units', min_value=32, max_value=128, step=32),
            activation='relu'
        ), 
        tf.keras.layers.Dropout(rate=hp.Float('dropout', min_value=0.2, max_value=0.5, step=0.1)),
        tf.keras.layers.Dense(4, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

tuner = kt.Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=5,
    factor=3,
    directory='my_dir',
    project_name='classification_optimization'
)

def run_tuner(X, y, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for train_index, val_index in kf.split(X):
        X_train_fold, X_val_fold = X[train_index], X[val_index]
        y_train_fold, y_val_fold = y[train_index], y[val_index]
        tuner.search(X_train_fold, y_train_fold, epochs=10, validation_data=(X_val_fold, y_val_fold))
    
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"""
        A pesquisa de hiperparâmetros foi concluída. O número ideal de dimensões de incorporação é {best_hps.get('embedding_dim')},
        o número ideal de unidades LSTM é {best_hps.get('lstm_units')}, e
        o número ideal de unidades densas é {best_hps.get('dense_units')},
        e a taxa de abandono ideal é {best_hps.get('dropout')}.
    """)
    return best_hps


best_hps = run_tuner(X_train, y_train)

final_model = tf.keras.Sequential([
    encoder,
    tf.keras.layers.Embedding(
        input_dim=len(encoder.get_vocabulary()),
        output_dim=best_hps.get('embedding_dim'),
        mask_zero=True
    ),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
        units=best_hps.get('lstm_units'),
        return_sequences=True
    )),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
        units=best_hps.get('lstm_units') // 2
    )),
    tf.keras.layers.Dense(
        units=best_hps.get('dense_units'),
        activation='relu'
    ),
    tf.keras.layers.Dropout(rate=best_hps.get('dropout')),
    tf.keras.layers.Dense(4, activation='softmax')
])


final_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

final_model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

y_pred = final_model.predict(X_test)
y_pred_classes = y_pred.argmax(axis=1)
conf_matrix = confusion_matrix(y_test, y_pred_classes)

disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=[0, 1, 2, 3])
disp.plot(cmap=plt.cm.Blues)
plt.show()

"""model = tf.keras.Sequential([
    encoder,
    tf.keras.layers.Embedding(
        input_dim=len(encoder.get_vocabulary()), #tamanho do vocabulário
        output_dim=64, 
        mask_zero=False
    ),
    # Permite que o modelo entenda o contexto dos dados de texto em ambas as direções (esquerda para direita e direita para esquerda), captando elementos tanto no início quanto no final das mensagens
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)), 
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(4, activation='softmax') # 4 classes na saída

])
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='sparse_categorical_crossentropy', metrics=['Accuracy'])"""
#epochs = 20
#history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test))

