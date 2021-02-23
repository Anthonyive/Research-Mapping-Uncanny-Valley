# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class FolksongsItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    songid = scrapy.Field()
    title = scrapy.Field()
    lyric = scrapy.Field()

class FolksongsArchiveItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    region = scrapy.Field()
    title = scrapy.Field()
    lyric = scrapy.Field()
    lyric_eng = scrapy.Field()