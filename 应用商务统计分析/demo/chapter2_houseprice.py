import urllib
from bs4 import BeautifulSoup
import re
import sys
import pandas as pd

pgs = 100
totResult = {}
for pg in range(1, pgs+1):
    print('-' * 100 + 'pg:%d' % pg)
    
    req = urllib.request.Request('http://gz.fang.lianjia.com/loupan/pg' + str(pg))
    response = urllib.request.urlopen(req)
    thepage = response.read()
    soup = BeautifulSoup(thepage,"lxml")
    
    result = {}
    for tag in soup.find_all(name="div", attrs={"class": re.compile("resblock-desc-wrapper")}):
        # resblock-name
        tag_name = tag.find_all(name="div", attrs={"class": re.compile("resblock-name")})
        lp_name = tag_name[0].find(name="a", attrs={"target": re.compile("_blank")}).text
        lp_type = tag_name[0].find(name="span", attrs={"class": re.compile("resblock-type")}).text
        sale_status = tag_name[0].find(name="span", attrs={"class": re.compile("sale-status")}).text

        # resblock-location
        tag_name = tag.find_all(name="div", attrs={"class": re.compile("resblock-location")})
        location0 = [i.text for i in tag_name[0].findAll(name="span")] # 行政区+（未知）
        location2 = tag_name[0].find(name="a", attrs={"target": re.compile("blank")}).text #明细地址

        # resblock-room
        tag_name = tag.find_all(name="a", attrs={"class": re.compile("resblock-room")})
        room = [i.text.replace('室', '') for i in tag_name[0].findAll(name="span") ]

        # resblock-area
        tag_name = tag.find_all(name="div", attrs={"class": re.compile("resblock-area")})
        area = tag_name[0].find(name="span").text.replace('建面 ', '').replace('㎡', '').split('-')

        # resblock-tag
        tag_name = tag.find_all(name="div", attrs={"class": re.compile("resblock-tag")})
        lp_tag = [i.text for i in tag_name[0].findAll(name="span")]

        # resblock-price
        tag_name = tag.find_all(name="div", attrs={"class": re.compile("resblock-price")})
        price = [i.text.replace(u"\xa0", "") for i in tag_name[0].find_all(name="div", attrs={"class": re.compile("main-price")})[0].findAll(name="span")]    

        # result
        result[lp_name] = {'type': lp_type
                           , 'sale_status': sale_status
                           , 'location0': location0[0]
                           , 'location1': location0[-1]
                           , 'location2': location2
                           , 'room': room
                           , 'room_min': min(room) if room != [] else 0
                           , 'room_max': max(room) if room != [] else 0
                           , 'room_num': len(room)
                           , 'area': area
                           , 'area_min': area[0]
                           , 'area_max': area[-1]
                           , 'tag': lp_tag
                           , 'tag_num': len(lp_tag)
                           , 'price': price[0]
                           , 'price_unit': price[-1]
                          }
        totResult = dict(totResult, **result)
        

df = pd.DataFrame(totResult).T
df['name'] = df.index
df.to_csv(u'houseprice_gz.csv', index=False)

