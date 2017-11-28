import pandas
import math
def getData():
	data=pandas.read_csv("winemag-data-130k-v2.csv")
	data=data.dropna(subset=['price'])
	data=data[data.price<200]
	variety = set(data['variety'])
	dic_variety = {r: i for r,i in zip(variety, range(len(variety)))}
	varietyList = [dic_variety[x] for x in data["variety"]]

	region = set(data['region_1'])
	dic_region = {r: i for r,i in zip(region, range(len(region)))}
	regionList = [dic_region[x] for x in data["region_1"]]

	province = set(data['province'])
	dic_province = {r: i for r,i in zip(province, range(len(province)))}
	provinceList = [dic_province[x] for x in data["province"]]

	winery = set(data['winery'])
	dic_winery = {r: i for r,i in zip(winery, range(len(winery)))}
	wineryList = [dic_winery[x] for x in data["winery"]]


	X=[[a,b,c,d] for a,b,c,d in zip(varietyList,regionList,provinceList,wineryList)]
	Y=data['price']
	temp=[y for y in Y if math.isnan(y)]
	print(len(temp),temp)
	return X,Y
