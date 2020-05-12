import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn import linear_model
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import bartlett
from scipy.stats import levene
from statsmodels.tsa.stattools import acf
from scipy.stats import jarque_bera
from scipy.stats import normaltest
import warnings
warnings.filterwarnings('ignore')

hava_durumu = pd.read_csv("weatherHistory.csv")

##########################################################################################################
# Önceki derste olduğu gibi, hedef değişkeninizin görünür sıcaklık ve sıcaklık arasındaki fark olduğu bir
# doğrusal regresyon modeli oluşturun. Açıklayıcı değişkenler olarak nem ve rüzgar hızı kullanın. Şimdi,
# modelinizi OLS kullanarak tahmin edin. R-kare ve ayarlanmış R-kare değerleri nelerdir?
# Tatminkar olduklarını düşünüyor musunuz? Açıklayın.

# hava_durumu["hedef_degisken"] = hava_durumu["Sicaklik"]-hava_durumu["gorunur_sicaklik"]
#
# Y = hava_durumu["hedef_degisken"]
# X = hava_durumu[["Nem","RuzgarHizi"]]
#
# lrm = linear_model.LinearRegression()
# lrm.fit(X, Y)
#
# print('Değişkenler: \n', lrm.coef_)
# print('Sabit değer (bias): \n\n\n', lrm.intercept_)
#
# X = sm.add_constant(X)
# results = sm.OLS(Y, X).fit()
# print(results.summary())

## Adj R-squared değeri 0.288. Bizim için kesinlikle yeterli olmayan bir değer. Daha fazla özellik ekleyip
## Bu değeri arttırmamız gerekiyor.


###############################################################################################################
#Daha sonra, yukarıdaki modele nem ve rüzgar hızı etkileşimini dahil edin ve OLS'yi kullanarak modeli tahmin edin.
# Şimdi, bu modelin R-kare değeri nedir? Bu model bir öncekine göre gelişti mi?

# hava_durumu["hedef_degisken"] = hava_durumu["Sicaklik"]-hava_durumu["gorunur_sicaklik"]
#
# hava_durumu["nem_RuzgarHızı_iliskisi"] = hava_durumu["Nem"] * hava_durumu["RuzgarHizi"]
#
# Y = hava_durumu["hedef_degisken"]
# X = hava_durumu[["Nem","RuzgarHizi","nem_RuzgarHızı_iliskisi"]]
#
# lrm = linear_model.LinearRegression()
# lrm.fit(X, Y)
#
# print('Değişkenler: \n', lrm.coef_)
# print('Sabit değer (bias): \n\n\n', lrm.intercept_)
#
# X = sm.add_constant(X)
# results = sm.OLS(Y, X).fit()
# print(results.summary())

# Adj. R-squared değeri 0.341 e yükseldi. Verimizi bir önceki özelliklere göre daha iyi açıklayabiliyoruz.
# Ama hala kesinlikle yeterli değil.


#################################################################################################################
# İlk modele ek açıklayıcı değişken olarak görünürlük ekleyin ve tahmin edin. R-kare arttı mı?
# Ayarlanmış R-kare değeri ne oldu? Tabloda ortaya çıkan farklılıkları, ayarlanmış R-kare içindeki
# iyileşme açısından etkileşim terimi ve görünürlük ile karşılaştırın. Hangisi daha kullanışlı?

# hava_durumu["hedef_degisken"] = hava_durumu["Sicaklik"]-hava_durumu["gorunur_sicaklik"]
#
# Y = hava_durumu["hedef_degisken"]
# X = hava_durumu[["Nem","RuzgarHizi","gorunurluk"]]
#
# lrm = linear_model.LinearRegression()
# lrm.fit(X, Y)
#
# print('Değişkenler: \n', lrm.coef_)
# print('Sabit değer (bias): \n\n\n', lrm.intercept_)
#
# X = sm.add_constant(X)
# results = sm.OLS(Y, X).fit()
# print(results.summary())

# Adj R-squared değeri ilk modele göre arttı. İlk modelde 0.288 iken şu an 0.303. Hala yeterli değil.
# İlk model mi bu model mi sorusunun cevabı ise hala yeterli olmamakla beraber bu model.


#################################################################################################################
#AIC ve BIC puanlarına göre yukarıdaki üç modelden en iyisini seçin.
# Mentor ile gerekçenizi tartışarak seçiminizi doğrulayın.

# Birici model için AIC ve BIC : 3.409 10**5 = 340,900
# İkinci model için AIC ve BIC : 3.334 10**5 = 333,400
# Üçüncü model için AIC ve BIC : 3.338 10**5 = 333,800

# AIC ve BIC için düşük değer daha iyi olduğuna göre, ikinci modeli seçebiliriz.


#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
# Ev fiyatları modelinizi tekrar çalıştırın ve F-testi, R-kare, ayarlanmış R-kare, AIC ve BIC
# kullanarak modelinizin uygunluğunu değerlendirin.

house= pd.read_csv("train.csv")

house['yeni_mi'] = np.where(house['YearBuilt']>=2005, 1, 0)

Y = house["SalePrice"]
X = house[["BedroomAbvGr","yeni_mi","FullBath","GarageCars","WoodDeckSF","OverallQual","LotArea"]]

X = sm.add_constant(X)
results = sm.OLS(Y, X).fit()
print(results.summary())

# F testi sonucu 529. R-squared değeri: 0.716 ve Adj. R-squared değeri 0.714. Değerlerimiz yeterli seviyede.
# R-squared ve Adj. R-squared değerleri birbirine çok yakın olduğu için verimli özellikleri seçtiğimizi söylebiliriz.
# AIC ve BIC değerlerini iki modeli karşılaştırırken kullanırız. Zaten tek modelimiz var. Düşük değerli AIC ve BIC
# daha iyidir.


#####################################################################################################################
#Modelinizin tatmin edici olduğunu düşünüyor musunuz? Açıklayın

# Adj. R-squared değerimiz 0.714. Yeterince yüksek bir değerde. 0.850 değerlerine kadar çekebiliriz ama 0.714'nin
# işimizi görebileceğini düşünüyorum. Daha fazlası overfitting'e kaçabilir.


#####################################################################################################################
#Modelinizin uygunluğunu iyileştirmek için, bazı değişkenleri ekleyerek veya kaldırarak farklı
# model özelliklerini deneyin.

# house= pd.read_csv("train.csv")
#
# house['yeni_mi'] = np.where(house['YearBuilt']>=2005, 1, 0)
# house["MasVnrArea"].fillna(0 , inplace=True)
#
# Y = house["SalePrice"]
# X = house[["BedroomAbvGr","yeni_mi","FullBath","GarageCars","WoodDeckSF","OverallQual",
#            "LotArea","MSSubClass","MasVnrArea","BsmtFinSF1","Fireplaces"]]
#
# X = sm.add_constant(X)
# results = sm.OLS(Y, X).fit()
# print(results.summary())

## Adj. R-squared değeri 0.766 ya yükseldi. Adj. R-squared değerini bir yerden sonra arttırmak çok zor.


#####################################################################################################################
#Denediğiniz her model için, uygun metrikleri alın ve modellerinizi birbiriyle karşılaştırın.
# Hangi model en iyisidir ve neden?

# Ilk modelimizin F değeri:  521.9                         Ikıncı modelimizin F değeri: 434.7
#                 Adj. R-squared:  0.714                                      Adj. R-squared: 0.766
#                 AIC:  3.52 e4                                               AIC: 3.49 e4
#                 BIC:  3.53 e4                                               BIC  3.50 e4

# Bütün değerlerimiz ikinci modelimizi gösteriyor.Düşük F değeri, büyük Adj. R-squared değeri , küçük AIC ve BIC değeri.











