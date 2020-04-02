# 鐵達尼號生存率 —預測及視覺化
### Titanic survival data prediction and visualization :

<div align="center"><img src="https://i.loli.net/2020/04/02/9ThUmy64CipwaV7.jpg" width='690' height='330'></div>

#### 分析目的：視覺化分析乘客的特徵，找出與存活率有關的變數，並且利用這些變數預測乘客的生存機率。
#### 分析流程：<br>
1. 觀察變數是否與存活率有關 
2.（處理 NA 值）
3. 特徵工程 （ex: 類別變數轉化成數值）
4. 將變數納入模型，進行預測 
5. 評估預測準確率，觀察變數是否對模型有幫助
#### 使用工具：sklearn, matplotlib, seaborn
#### 資料大小：
* Trainning Data : 891 rows * 12 columns (60 KB)
* Testing Data : 418 rows * 11 columns (28 KB)
#### [資料來源](https://www.kaggle.com/c/titanic) 
#### 程式碼:  
* [預測與特徵工程](https://github.com/albert90162/Titanic/blob/master/Prediction.ipynb) 
* [視覺化分析與 EDA ](https://github.com/albert90162/Titanic/blob/master/EDA%20and%20Visualizations.ipynb)
    
<br>

## 目錄：
1. [性別與艙等 Sex & Pclass](#sex--pclass)
2. [票價 Fare](#fare)
3. [乘客之間的關係 Connection](#connection)
4. [年齡 Age](#age)
5. [結論](#結論)
<br>
<br>

## Sex & Pclass:<br>
> 想到生存率，較直觀的想法就是與年齡、性別、艙等有關，而年齡有 NA 值，因此先從性別與艙等著手 <br>
<br>

### Sex:<br>
對性別與生存率作圖：
<div align="center"><img src="https://i.loli.net/2020/04/02/w3mlcCKioZySsvz.png" width='500' height='330'></div>
從圖中可發現，男女之間的生存率有明顯的不同。<br><br>
<div align="center"><img src="https://i.loli.net/2020/04/02/skF9EB6cKJG5Mfj.jpg" width='300' height='250'></div>
深入觀察可發現：女性的存活率約 75%，而男性的存活率卻不到 20%
<br>
<br>

### Pclass:<br>
對艙等與生存率作圖：
<div align="center"><img src="https://i.loli.net/2020/04/02/MzFHPZkyugSE6wm.png" width='600' height='400'></div>
從圖中可發現，三個艙等的生存率也有明顯的不同，尤其是三等艙的存活率明顯更低。<br>
<br>

<div align="center"><img src="https://i.loli.net/2020/04/02/7a8jZB5S1eo2svU.jpg" width='300' height='250'></div>
深入觀察可發現：頭等艙的生存率有 63%，二等艙的生存率則有 47%，三等艙的生存率只有 24%。
<br><br>

**因此，性別與艙等明顯都與生存率有關係。單獨以這兩個特徵來預測生存率看看：**<br>
>預測前先進行資料處理：<br>1. 將 Sex 中的女性設為0，男性設為1 <br>2. 將資料切分成訓練及測試資料 <br> 3. 將訓練資料的 Survived 和 PassengerId 移除  

<br>

建立隨機森林模型，並使用性別與艙等進行預測後，訓練資料中的正確率為 73.1%。
*此時將預測結果提交至 Kaggle 的正確率則為 76%*<br>
#### 小結：顯然目前的正確率還不夠，雖然找到與生存率高度相關的特徵，但是還需要逐一考慮其他變數，以達到更高的正確率
<br><br>

## Fare:<br>
> 接下來觀察票價與生存率之間是否也有關係<br><br>

首先將票價取 log，並觀察 log Fare 對艙等和生存率的關係：
<div align="center"><img src="https://i.loli.net/2020/04/02/a6qWHdXyfIN8nSw.png" width='690' height='330'></div>
從圖中可看出，在頭等艙跟二等艙中，生存者的票價相較於罹難者高一些（尤其是在頭等艙）。而在三等艙的分布狀況則差不多。<br><br>
<div align="center"><img src="https://i.loli.net/2020/04/02/fMscUzxmNZ8SgOl.jpg" width='350' height='300'></div>
深入觀察可發現：以中位數而言，三個艙等的生存者所付的票價確實較罹難者高，只是三等艙的生存者與罹難者所支付的票價幾乎相同。這應證了上圖的觀察。<br>

**綜上所述，不論在哪個艙等，生存者確實有支付較多票價的趨勢，因此將票價納入預測考量的變數之中**

<br><br>

### 但是，票價要如何幫助預測生存率？
首先，Fare 有一個 NA值，在這裡直接用中位數代入<br>
接著，將票價分群，但是不知道要分幾群較能準確預測，因此按照比例分別嘗試分成4、5、6群，將三種分群結果轉成 3 個columns，加入訓練資料，並觀察哪種分群較能有效區隔出各群之間的差異<br>
<br>

將三種分法的票價分別對生存率作圖：
<div align="center"><img src="https://i.loli.net/2020/04/02/XSYbB37pIKOPCz2.png" width='690' height='330'></div>
<br>

**從圖中看起來是分成 4 群會有最明顯的差異，但是無法確定分成 4 群對生存率的預測最佳，因此進行特徵選擇：**
>資料處理：保留訓練資料的 [性別、艙等、分四群結果、分五群結果、分六群結果]，其他變數則捨棄。並使用 RFECV 進行特徵選擇

<br>

選擇後的結果為這五種特徵都應該保留，且五種特徵得到的準確率也都相近，無法看出要留下哪個特徵，因此進一步對三種分群方式進行 cross-validation，並分別對準確率作圖:
<div align="center"><img src="https://i.loli.net/2020/04/02/y356DxNSbOQAgYj.png" width='690' height='330'></div>
看起來分成5、6群的準確率高於分成4群，而6群又比5群的準確率高一些。

#### 小結：隨後將三種分群方式分別上傳到 Kaggle 後，發現分成 5 群的準確率反而最高，推斷可能分成 6 群會發生 overfitting 的現象，因此在這邊採用 5 群區分。所以目前保留的特徵有性別、艙等、分成五群的票價，這三個特徵。

<br><br>

## Connection:<br>
> Connection 表示乘客之間可能彼此有關係，例如朋友或家人。而直觀上會認為有 connection 的人們可能會一起存活或一起罹難，因此深入觀察 connection
<br>
<div align="center"><img src="https://i.loli.net/2020/04/02/6tyOgk4RcL23x8M.jpg" width='300' height='250'></div>

使用 Ticket 來尋找乘客之間的 connection，先看訓練資料中的 Ticket 資訊：沒有 NA 值，共 891 筆，但其中只有 681 個 unique 值，代表有許多乘客持相同的 Ticket 上船，而這些乘客之間有可能是朋友或家人的關係
隨後將所有資料持有相同 Ticket 的乘客分群後，發現同一群的乘客通常會一起存活或罹難，符合一開始的假設，表示 connection 確實有可能影響存活率
<br>
<br>
接下來觀察各群組的 family_size ，將有 connection 的乘客，family_size > 1 的歸類為家庭；family_size = 1 的歸類為朋友。
可以發現共有 596 名乘客與其他乘客持有和其他乘客相同的票根，其中有 127 位乘客之間是朋友關係，469 位乘客之間是家人關係。
<br>
<br>
隨後產生新的變數 Connection_survival，其定義為**除了自己以外**，群組中有人生還則為 1 ；群組中無人生還則為 0 ；群組中其他人皆為NaN，或沒有 connection 的，定義為 0.5
>舉例來說，假設有個3人群組，A存活，BC死亡，則 A 的 Connection_survival 為 0，B 和 C 的 Connection_Survival 則為 1
<br>
<div align="center"><img src="https://i.loli.net/2020/04/02/49rVqxSF3Q1syLI.jpg" width='800' height='350'></div>
<br>

此變數的目的，在於觀察 connection 的存在是否真的會影響生存率，因此計算 0、0.5、1 三種 Connection_Survival 的實際生存率：
* Connection_Survival 為 0 的人們，實際生存率為 22%
* Connection_Survival 為 0.5 的人們，實際生存率為 30%
* Connection_Survival 為 1 的人們，實際生存率為 72%
<div align="center"><img src="https://i.loli.net/2020/04/02/X67lE8xDAYZvuH3.jpg" width='300' height='250'></div>
<br>
這項數據，代表當某一乘客所在的群組中，其他人皆罹難，則該名乘客本身的存活率也較低；當群組中有人生還，則該名乘客的存活率較高。<br>
而 0.5 則表示獨自上船或與其他乘客沒有 connection 的乘客，這群人的存活率為 30%
<br><br>

透過以上的觀察，Connection_Survival 明顯與存活率有關係，因此將 Connection_Survival 納入考量的變數，並且進行預測後，**在訓練資料中可得 82% 的正確率；在 Kaggle (預測資料) 可得 80.3% 的正確率**
<br><br>

#### 小結：connection 確實也對預測存活率有幫助。目前模型中存在的特徵有：性別、艙等、票價分群（5群）、Connection_Survival


<br><br>

## Age:<br>
>相較於 Fare 的 NA 數量，Age 有 263 個 NA ，因此如何處理這些 NA 將會成為能否使用 Age 這項特徵的關鍵
<br>
首先，觀察這兩百多筆 NA 值來自哪裡：
<div align="center"><img src="https://i.loli.net/2020/04/02/DzfUE73Otu4FsWR.png" width='650' height='350'></div>
從性別而言，男女擁有 NA 值的比例差不多；但以艙等而言，大部分的 NA 值來自三等艙。
<br><br>

因此，先確認年齡是否會影響頭等與二等艙的存活率：
<div align="center"><img src="https://i.loli.net/2020/04/02/ERfBGvNXa1MF9CY.png" width='690' height='400'></div>
從圖中可看出，左邊有一塊突出的藍色，代表存活率較高，約在小於 16 歲以前；而大於 16 歲後，年齡與存活率之間就沒有顯著的關係，因此只將年齡區分為 [>16 , <16] 兩個族群（至於 70 歲以上，由於樣本數太少，因此同樣不納入考量）           
<br>
<br>

回到缺失值的部分，在此選擇以姓名當中的稱謂來填補年齡的缺失值：<br>
稱謂主要以 Mr、Master、Miss、Mrs 為主，其餘名稱以'Rare'代替，並計算五類稱謂的年齡中位數
* Mr : 29
* Rare : 47
* Master : 4
* Miss : 22
* Mrs : 36
<br>
隨後將此中位數填入對應稱謂的 NA 值，便完成 NA 值的處理。

<br>
最後，新增一個類別變數 Ti_minor，將年紀區分為 >16 歲和 < 16 歲，並將 Ti_minor 納入考量的變數並進行預測後，
<br>

**訓練資料的準確率可到 84.6% ，而 Kaggle 上的正確率則可到 82.3%**
<br>

#### 小結：考量了年齡後，正確率再提升了 2%，而此時在 Kaggle 上已經可以達到 Top 3% 的排名了。

<br>
<br>

## 結論：<br>
在資料中提供的 12 項變數中，扣除要預測的變數 Survived 以及乘客ID後，使用了 [ 性別、艙等、票價、Connection、年紀 ] 等五項變數進行預測及建模，並且在訓練及測試資料都得到超過 80% 的正確率，說明了這五項變數皆能有效地預測存活率，且沒有 overfitting 的情形發生。           
<br>
<br>
<br>

*****

#### The End 
#### 作者：鮑威宇, Albert Pao 


