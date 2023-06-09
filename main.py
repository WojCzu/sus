import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, cross_val_predict, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

'''
"matplotlib.pyplot" jest używane do wykreślania wyników i tworzenia wykresów.
"numpy "jest używane w celu obliczenia odchylenia standardowego i średniej, które są wykorzystywane do usunięcia wartości odstających.
"pandas" jest używane do manipulacji danymi oraz do wczytania pliku CSV.
"sklearn.linear_model (Lasso, Ridge)" jest używane do stworzenia modeli Lasso i Ridge.
"sklearn.model_selection (train_test_split, GridSearchCV, cross_val_score, cross_val_predict, KFold)" jest wykorzystywane do podziału danych na zestawy treningowe i testowe, strojenia hiperparametrów, przeprowadzania walidacji krzyżowej oraz przewidywania.
"sklearn.preprocessing (StandardScaler)" jest używane do przeskalowania cech do rozkładu normalnego (średnia = 0, odchylenie standardowe = 1), co jest wymagane przez niektóre algorytmy ML.
"sklearn.pipeline (Pipeline)" jest używane do łączenia kolejnych etapów przetwarzania i estymacji modelu.
'''

file = pd.read_csv('Houses.csv', sep=',')
df = pd.DataFrame(file)

'''
num - numer wiersza
address - adres nieruchomości
city - miasto, w którym znajduje się nieruchomość
floor - piętro, na którym znajduje się nieruchomość
id - identyfikator nieruchomości
latitude - szerokość geograficzna nieruchomości
longitude - długość geograficzna nieruchomości
price - cena nieruchomości
rooms - liczba pokoi w nieruchomości
sq - powierzchnia nieruchomości (w metrach kwadratowych)
year - rok budowy lub remontu nieruchomości
'''


def preprocessing(data):

    # 1. Usuwanie niepotrzebnych danych
    data = data.drop(['num', 'address', 'city', 'id', 'rooms'], axis=1)

    # 2. Usuwanie wierszy z brakującymi danymi
    data = data.dropna()


    # 3. Usuwanie skrajnych cen, które odbiegają od średniej o więcej niż trzy odchylenia standardowe
    data = data[np.abs(data["price"]-data["price"].mean())<=(3*data["price"].std())]

    # 4. Podział danych na cechy i etykiety
    y = data.price
    X = data.drop('price', axis = 1)

    return X, y

def model(pipeline, parameters, X_train, y_train, X, y):

    # definiowanie obiektu GridSearchCV
    grid_obj = GridSearchCV(estimator=pipeline,    # pipeline dla którego będą przeszukiwane parametry
                            param_grid=parameters, # słowniyk zawierający parametry modelu
                            cv=3,                  # liczba podziałów w walidacji krzyżowej
                            scoring='r2',          # metryka do oceny jakości modelu
                            verbose=2,             # jak wiele informacji na temat procesu ma być wyświetlane
                            n_jobs=1,              # ile rdzeni procesora ma być użyte do obliczeń
                            refit=True)            # po zakończeniu przeszukiwania, model jest ponownie trenowany na całym zestawie danych treningowych
    
    # dopasowanie danych treningowych
    grid_obj.fit(X_train, y_train)

    # tworzymy DataFrame zawierający wyniki przeszukiwania siatki 
    results = pd.DataFrame(grid_obj.cv_results_)
    # sortujemy wyniki względem średniego wyniku testowego
    results_sorted = results.sort_values(by=['mean_test_score'], ascending=False)

    print("##### Wyniki")
    print(results_sorted)

    #Wydrukowujemy również indeks, wynik i parametry dla najlepszego modelu.
    print("best_index", grid_obj.best_index_)
    print("best_score", grid_obj.best_score_)
    print("best_params", grid_obj.best_params_)

    '''
    Definiujemy estymator jako najlepszy estymator z przeszukiwania siatki,
    następnie wykonujemy walidację krzyżową na danych X i y, wykorzystując do tego 5 podziałów danych z losowym mieszaniem.
    Średni wynik walidacji krzyżowej jest następnie wydrukowany.
    '''
    estimator = grid_obj.best_estimator_
    shuffle = KFold(n_splits=5,
                    shuffle=True,
                    random_state=0)
    cv_scores = cross_val_score(estimator,
                                X,
                                y.values.ravel(),
                                cv=shuffle,
                                scoring='r2')
    print("##### CV Results")
    print("mean_score", cv_scores.mean())

    # współczynniki modelu lub ważności cech
    print("Model coefficients: ", list(zip(list(X), estimator.named_steps['clf'].coef_)))


    '''
    Tworzymy wykres punktowy rzeczywistych wartości w stosunku do przewidzianych wartości,
    dodajemy linie identyczności,
    dodajemy adnotacje z wynikiem R^2 i najlepszymi parametrami,
    a następnie wyświetlamy wykres.
    '''

    y_pred = cross_val_predict(estimator, X, y, cv=shuffle)

    plt.scatter(y, y_pred)
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    plt.plot([xmin, xmax], [ymin, ymax], "g--", lw=1, alpha=0.4)
    plt.xlabel("Rzeczywiste ceny")
    plt.ylabel("Przewidywane ceny")
    plt.annotate(' R-squared CV = {}'.format(round(float(cv_scores.mean()), 3)), size=9,
             xy=(xmin,ymax), xytext=(10, -15), textcoords='offset points')
    plt.annotate(grid_obj.best_params_, size=9,
                 xy=(xmin, ymax), xytext=(10, -35), textcoords='offset points', wrap=True)
    plt.title('Przewidywane ceny vs. Rzeczywiste ceny')
    plt.show()

# utworzenie pipeline, który pozwala na sekwencyjne wykonywanie operacji przetwarzania danych i modelowania (scl = scaler, clf = classifier)
# przetwarzanie danych przy użyciu StandardScaler w celu ich normalizacji
# modelowanie, wykorzystujące model regresji Lasso/Ridge

pipe_lasso = Pipeline([('scl', StandardScaler()),
           ('clf', Lasso(max_iter=50000))])
pipe_ridge = Pipeline([('scl', StandardScaler()),
           ('clf', Ridge())])

#  słownik, który zostanie użyty do dostrajania hiperparametrów modelu regresji w ramach procedury walidacji krzyżowej
param_lasso = {'clf__alpha': [0.01, 0.1, 1, 5, 10]}
param_ridge = {'clf__alpha': [0.01, 0.1, 1, 5, 10]}

# Wstępne przetwarzanie danych
X, y = preprocessing(df)

# Podział danych na dane treningowe (95%) i testowe (5%) z podziałem deterministycznym (random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=0)

# Uruchomienie modelu i walidacji
model(pipe_lasso, param_lasso, X_train, y_train, X_test, y_test)
model(pipe_ridge, param_ridge, X_train, y_train, X_test, y_test)