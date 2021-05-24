import numpy as np
import pandas as pd
import random
import math
import matplotlib.pyplot as plt

# Global variables
phase = "train"  # phase can be set to either "train" or "eval"
highbrand = []
midbrand = []
lowbrand = []

"""
You are allowed to change the names of function "arguments" as per your convenience,
but it should be meaningful.

E.g. y, y_train, y_test, output_var, target, output_label, ... are acceptable
but abc, a, b, etc. are not.

"""
def feature_return(file_path, return_df = False):
    ## read file
    df = pd.read_csv(file_path)

    #Splitting brand and model name from the name column and using them instead to reduce overfitting
    df_brand = pd.DataFrame(df.name.apply(lambda x : x.split()[:2]).tolist(), columns=['brand', 'model'])
    df = pd.concat([df, df_brand], axis = 1)
    df = df.drop(columns='name')

    ##Creating dummies for fuel, transmission, seller and owner
    df_fuel = pd.get_dummies(df.fuel, prefix = 'fuel')
    df_trans = pd.get_dummies(df.transmission, prefix = 'trans')
    df_seller = pd.get_dummies(df.seller_type, prefix = 'seller')
    df_owner = pd.get_dummies(df.owner, prefix = '')
    df = pd.concat([df, df_seller, df_owner, df_trans, df_fuel], axis = 1)


    ## Replacing engine and mileage columns with actual values rather than strings
    for i in range(len(df)):
        if isinstance(df.loc[i, 'engine'], str) :
            engine = df.loc[i, 'engine'].split()[0]
            engine_no = float(engine)
            df.loc[i,'engine'] = engine_no
        if isinstance(df.loc[i, 'mileage'], str) :
            mileage_no = float(df.loc[i, 'mileage'].split()[0])
            mileage_unit = df.loc[i, 'mileage'].split()[1]
            if mileage_unit == 'km/kg':
                if df.loc[i, 'fuel'] == 'Diesel':
                    df.loc[i, 'mileage'] = mileage_no*0.84
                elif df.loc[i, 'fuel'] == 'Petrol':
                    df.loc[i, 'mileage'] = mileage_no*0.74
                elif df.loc[i, 'fuel'] == 'LPG':
                    df.loc[i, 'mileage'] = mileage_no*0.51
                elif df.loc[i, 'fuel'] == 'CNG':
                    df.loc[i, 'mileage'] = mileage_no*0.714
            else:
                df.loc[i, 'mileage'] = mileage_no


    ## Replacing torque with actual values rather than strings
    for i in range(len(df)):
        torque = -1
        first_word = None
        second_word = None
        if isinstance(df.loc[i, 'torque'], str) :
            replacedword = df.loc[i, 'torque']
            if 'at' in df.loc[i, 'torque']:
                replacedword = df.loc[i, 'torque'].replace('at', '@')
            wordlist = replacedword.split('@')
            first_word = wordlist[0]
            if len(wordlist) > 1:
                second_word = wordlist[1]
            if 'Nm' in first_word and 'kgm' in first_word:
                first_word = first_word.split('Nm')[0]
                torque = float(first_word)
            elif 'kgm' in first_word or 'KGM' in first_word or (isinstance(second_word, str) and ('kgm' in second_word or 'KGM' in second_word)):
                if ((isinstance(second_word, str)) and ('kgm' in second_word or 'KGM' in second_word)):
                    torque = float(first_word)*9.81
                elif 'kgm' in first_word:
                    first_word = first_word.split('kgm')[0]
                    torque = float(first_word)*9.81
                elif 'KGM' in first_word:
                    first_word = first_word.split('KGM')[0]
                    torque = float(first_word)*9.81
            elif 'Nm' in first_word or 'nm' in first_word or 'NM' in first_word or (isinstance(second_word, str) and ('Nm' in second_word or 'nm' in second_word or 'NM' in second_word)):
                if 'Nm' in first_word:
                    first_word = first_word.split('Nm')[0]
                    torque = float(first_word)
                elif 'nm' in first_word:
                    first_word = first_word.split('nm')[0]
                    torque = float(first_word)
                elif 'NM' in first_word:
                    first_word = first_word.split('NM')[0]
                    torque = float(first_word)
                elif isinstance(second_word, str) and ('Nm' in second_word or 'nm' in second_word or 'NM' in second_word):
                    torque = float(first_word)
            if torque == -1:
                df.loc[i, 'torque'] = np.nan
            else:
                df.loc[i, 'torque'] = torque

    ## Creating a DataFrame of only not null values to assign relevant means to the null valued rows
    df_corrector = df[(df['engine'].notnull())]

    for i in range(len(df)):
        if np.isnan(df.loc[i,'engine']):
            brand = df.loc[i, 'brand']
            model = df.loc[i, 'model']
            df_temp = df_corrector[(df_corrector['brand'] == brand) & (df_corrector['model'] == model)]
            df.loc[i, 'engine'] = df_temp['engine'].mean()
            df.loc[i, 'max_power'] = df_temp['max_power'].mean()
            df.loc[i, 'seats'] = df_temp['seats'].mean()
            df.loc[i, 'mileage'] = df_temp['mileage'].mean()
        if np.isnan(df.loc[i, 'engine']):
            df.loc[i, 'engine'] = df_corrector['engine'].mean()
            df.loc[i, 'max_power'] = df_corrector['max_power'].mean()
            df.loc[i, 'seats'] = df_corrector['seats'].mean()
            df.loc[i, 'mileage'] = df_corrector['mileage'].mean()

        if np.isnan(df.loc[i, 'torque']):
            brand = df.loc[i, 'brand']
            model = df.loc[i, 'model']
            df_temp = df_corrector[(df_corrector['brand'] == brand) & (df_corrector['model'] == model)]
            df.loc[i, 'torque'] = df_temp['torque'].mean()
        if np.isnan((df.loc[i, 'torque'])):
            df.loc[i, 'torque'] = df_corrector['torque'].mean()

    ## Classifying brands on the basis of their average selling price
    if phase == 'train':
        for i in range(len(df)):

            brand = df.loc[i, 'brand']
            df_temp = df[(df['brand'] == brand)]
            sellprice = df_temp['selling_price'].mean()
            if len(df_temp) <= 3:
                brand = 'other'
            elif sellprice < 500000:
                if brand not in lowbrand:
                    lowbrand.append(brand)
                brand = 'low'
            elif sellprice < 1500000:
                if brand not in midbrand:
                    midbrand.append(brand)
                brand = 'mid'
            else:
                if brand not in highbrand:
                    highbrand.append(brand)
                brand = 'high'
            df.loc[i, 'brand'] = brand
    else:
        for i in range(len(df)):
            brand = df.loc[i, 'brand']
            if brand in highbrand:
                brand = 'high'
            elif brand in lowbrand:
                brand = 'low'
            elif brand in midbrand:
                brand = 'mid'
            else:
                brand = 'other'
            df.loc[i, 'brand'] = brand


    ## Creating brandclassifier dataframe to help get dummies for brands that are classified
    df_brandclassifier = pd.get_dummies(df.brand, prefix = 'brand')
    df = pd.concat([df, df_brandclassifier], axis = 1)

    if 'fuel_CNG' not in df:
        df['fuel_CNG'] = 0
    if 'fuel_Petrol' not in df:
        df['fuel_Petrol'] = 0
    if 'fuel_LPG' not in df:
        df['fuel_LPG'] = 0
    if 'fuel_Diesel' not in df:
        df['fuel_Diesel'] = 0
    if 'brand_high' not in df:
        df['brand_high'] = 0
    if 'brand_mid' not in df:
        df['brand_mid'] = 0
    if 'brand_low' not in df:
        df['brand_low'] = 0
    if 'brand_other' not in df:
        df['brand_other'] = 0
    if 'seller_Dealer' not in df:
        df['seller_Dealer'] = 0
    if 'seller_Individual' not in df:
        df['seller_Individual'] = 0
    if 'seller_Trustmark Dealer' not in df:
        df['seller_Trustmark Dealer'] = 0
    if 'trans_Manual' not in df:
        df['trans_Manual'] = 0
    if 'trans_Automatic' not in df:
        df['trans_Automatic'] = 0
    if '_First Owner' not in df:
        df['_First Owner'] = 0
    if '_Second Owner' not in df:
        df['_Second Owner'] = 0
    if '_Third Owner' not in df:
        df['_Third Owner'] = 0
    if '_Fourth & Above Owner' not in df:
        df['_Fourth & Above Owner'] = 0
    if '_Test Drive Car' not in df:
        df['_Test Drive Car'] = 0

    ## Dropping useless columns after obtaining the phi matrix
    if 'selling_price' in df:
        ydf = df['selling_price']
        df = df.drop(columns = 'selling_price')
    else:
        ydf = df['Index']
    df = df.drop(columns = 'Index')
    df = df.drop(columns = 'fuel')
    df = df.drop(columns = 'seller_type')
    df = df.drop(columns = 'transmission')
    df = df.drop(columns = 'owner')
    df = df.drop(columns = 'brand')
    df = df.drop(columns = 'model')

    df["b Column"] = 1

    if(return_df == True):
        phi = df
        y = ydf
    else:
        phi = df.to_numpy()
        phi = np.array(phi, dtype = np.float)
        y_array = ydf.to_numpy()
        y = y_array.reshape((-1,1))
        y = np.array(y, dtype = np.float)

    return phi, y


def get_features(file_path):
    # Given a file path , return feature matrix and target labels
    df, ydf = feature_return(file_path, True)

    normalized_df=(df-df.min())/(df.max()-df.min())

    num_rows, num_cols = df.shape
    for i in range(num_cols):
        if (df.iloc[:, i] == 0).all() or normalized_df.iloc[:, i].isnull().values.all():
            normalized_df.iloc[:, i] = df.iloc[:, i]

    phi = normalized_df.to_numpy()
    phi = np.array(phi, dtype = np.float)
    y_array = ydf.to_numpy()
    y = y_array.reshape((-1,1))
    y = np.array(y, dtype = np.float)


    return phi, y

def get_features_basis(file_path):
    # Given a file path , return feature matrix and target labels
    df, ydf = feature_return(file_path, True)

    ## Also adding new feature mileage^2
    df['mileage'] = df['mileage'].pow(3)
    df = df.rename(columns= {'mileage' : 'mileage^2'})
    df['km_driven'] = df['km_driven'].pow(1.2)
    df = df.rename(columns= {'km_driven' : 'km_driven^1.2'})
    df['engine'] = df['engine'].pow(2)
    df = df.rename(columns= {'engine' : 'engine^2'})
    ## min max normalization of phi matrix to get max values of features as 1
    normalized_df=(df-df.min())/(df.max()-df.min())

    num_rows, num_cols = df.shape
    for i in range(num_cols):
        if (df.iloc[:, i] == 0).all() or normalized_df.iloc[:, i].isnull().values.all():
            normalized_df.iloc[:, i] = df.iloc[:, i]

    phi = normalized_df.to_numpy()
    phi = np.array(phi, dtype = np.float)
    y_array = ydf.to_numpy()
    y = y_array.reshape((-1,1))
    y = np.array(y, dtype = np.float)

    return phi, y

def compute_RMSE(phi, w , y) :
    # Root Mean Squared Error
    num_rows, num_cols = phi.shape
    y_expect = phi.dot(w)
    error = math.sqrt((1/num_rows)*sum(val**2 for val in (y - y_expect)))
    return error

def generate_output(phi_test, w):
    # writes a file (output.csv) containing target variables in required format for Submission.
    numrow, numcol = phi_test.shape
    X = np.arange(numrow)
    Y = np.matmul(phi_test, w)
    Y = Y.astype(np.float)
    df = pd.DataFrame({'Id': X})
    df['Expected'] = Y
    df.to_csv('output.csv', index = False)

    return None

def closed_soln(phi, y):
    # Function returns the solution w for Xw=y.
    return np.linalg.pinv(phi).dot(y)

def gradient_descent(phi, y, phi_dev, y_dev) :
    # Implement gradient_descent using Mean Squared Error Loss
    # You may choose to use the dev set to determine point of convergence
    num_rows, num_cols = phi.shape
    num_rows_dev, num_cols_dev = phi_dev.shape
    iterations = 1000
    learning_rate = 0.00004
    cost = []
    w = np.zeros((num_cols, 1))
    for i in range(iterations):
        y_expect = phi.dot(w)
        cost.append(sum(val**2 for val in (y - y_expect)))
        for j in range(num_cols):
            phi_2d = phi[:, j].reshape(-1, 1)
            slope = -(2)*sum(val[0] for val in np.multiply(phi_2d, y - y_expect))
            w[j][0] = w[j][0] - learning_rate*slope

    return w

def sgd(phi, y, phi_dev, y_dev) :
    # Implement stochastic gradient_descent using Mean Squared Error Loss
    # You may choose to use the dev set to determine point of convergence
    num_rows, num_cols = phi.shape
    num_rows_dev, num_cols_dev = phi_dev.shape
    iterations = 1000
    learning_rate = 0.0006
    cost = []
    w = np.zeros((num_cols, 1))
    for i in range(iterations):
        random_index = random.randint(0, num_rows-1)
        sample_x = phi[random_index].reshape(-1,1)
        sample_y = y[random_index]
        Y_expect = sample_x.T.dot(w)
        cost.append((sample_y - Y_expect)**2)
        for j in range(num_cols):
            phi_2d = sample_x[j].reshape(-1, 1)
            slope = (-(2)*np.multiply(phi_2d, sample_y-Y_expect))
            w[j][0] = w[j][0] - learning_rate*slope
        cost_dev = sum(val**2 for val in (y_dev - phi_dev.dot(w)))
        if cost_dev < 1e+10:
          break

    return w


def pnorm(phi, y, phi_dev, y_dev, p) :
    # Implement gradient_descent with p-norm regularisation using Mean Squared Error Loss
    # You may choose to use the dev set to determine point of convergence
    num_rows, num_cols = phi.shape
    num_rows_dev, num_cols_dev = phi_dev.shape
    Lamda = 0.2;
    iterations = 1000
    if p == 2:
        learning_rate = 0.000035
    elif p == 4:
        learning_rate = 0.00003
    cost = []
    cost_dev = []
    w = np.zeros((num_cols, 1))
    for i in range(iterations):
          y_expect = phi.dot(w)
          cost.append((sum(val**2 for val in (y - y_expect))) + Lamda * sum(val**p for val in w))
          for j in range(num_cols):
            phi_2d = phi[:, j].reshape(-1, 1)
            slope = ((-(2)*sum(val[0] for val in np.multiply(phi_2d, y - y_expect))) + p*Lamda*(w[j][0]**(p-1)))
            w[j][0] = w[j][0] - learning_rate*slope
          if i > 1 and abs(cost[i] - cost[i-1]) < 1e+08:
            break
          cost_dev.append((sum(val**2 for val in (y_dev - phi_dev.dot(w)))) + Lamda * sum(val**p for val in w))
          if abs(cost_dev[i] - cost_dev[i-1]) < 1e+11:
            break


    return w


def main():

    #The following steps will be run in sequence by the autograder.#

        ######## Task 1 #########
        phase = "train"
        phi, y = get_features('df_train.csv')
        phase = "eval"
        phi_dev, y_dev = get_features('df_val.csv')
        w1 = closed_soln(phi, y)
        w2 = gradient_descent(phi, y, phi_dev, y_dev)
        r1 = compute_RMSE(phi_dev, w1, y_dev)
        r2 = compute_RMSE(phi_dev, w2, y_dev)
        print('1a: ')
        print(abs(r1-r2))
        w3 = sgd(phi, y, phi_dev, y_dev)
        r3 = compute_RMSE(phi_dev, w3, y_dev)
        print('1c: ')
        print(abs(r2-r3))

        ######## Task 2 #########
        w_p2 = pnorm(phi, y, phi_dev, y_dev, 2)
        w_p4 = pnorm(phi, y, phi_dev, y_dev, 4)
        r_p2 = compute_RMSE(phi_dev, w_p2, y_dev)
        r_p4 = compute_RMSE(phi_dev, w_p4, y_dev)
        print('2: pnorm2')
        print(r_p2)
        print('2: pnorm4')
        print(r_p4)

        ######## Task 3 #########
        phase = "train"
        phi_basis, y = get_features_basis('df_train.csv')
        phase = "eval"
        phi_dev, y_dev = get_features_basis('df_val.csv')
        w_basis = pnorm(phi_basis, y, phi_dev, y_dev, 2)
        rmse_basis = compute_RMSE(phi_dev, w_basis, y_dev)
        print('Task 3: basis')
        print(rmse_basis)

def task5(w):
    numrow, numcol = w.shape
    secondmin = abs(w[0][0])
    min = abs(w[0][0])
    i = 0
    k = 0
    for j in range(numrow):
        if abs(w[j][0]) < min:
            secondmin = min
            min = abs(w[j][0])
            k = i
            i = j
        elif w[j][0] < secondmin:
            secondmin = abs(w[j][0])
            k = j
    print(k, i, w[k][0], w[i][0])

def split_data(phi, y, size):
    phi_split = phi[0:size-1][:]
    y_split = y[0:size-1]
    return phi_split, y_split

def task4():
    phase = "train"
    phi, y = get_features('df_train.csv')
    phase = "eval"
    phi_dev, y_dev = get_features('df_val.csv')
    phi_split, y_split = split_data(phi, y, 2500)
    w1 = gradient_descent(phi_split, y_split, phi_dev, y_dev)
    Y = []
    X = [2500, 3000, 3500, 4500]
    Y.append(compute_RMSE(phi_dev, w1, y_dev))
    phi_split, y_split = split_data(phi, y, 3000)
    w1 = gradient_descent(phi_split, y_split, phi_dev, y_dev)
    Y.append(compute_RMSE(phi_dev, w1, y_dev))
    phi_split, y_split = split_data(phi, y, 3500)
    w1 = gradient_descent(phi_split, y_split, phi_dev, y_dev)
    Y.append(compute_RMSE(phi_dev, w1, y_dev))
    w1 = gradient_descent(phi_split, y_split, phi_dev, y_dev)
    Y.append(compute_RMSE(phi_dev, w1, y_dev))
    plt.plot(X, Y)
    plt.show()
    print("Task 4:")
    print(X)
    print(Y)

main()


# Run task(4) function for task 4
#task4()

# Run task(5) function for task 5
#task5()


# Run below code for task 6
'''phase = "train"
phi, y = get_features_basis('df_train.csv')
phase = "eval"
phi_dev, y_dev = get_features_basis('df_val.csv')
phi_test, y_test = get_features_basis('df_test.csv')
w = pnorm(phi, y, phi_dev, y_dev, 2)
generate_output(phi_test, w)'''
