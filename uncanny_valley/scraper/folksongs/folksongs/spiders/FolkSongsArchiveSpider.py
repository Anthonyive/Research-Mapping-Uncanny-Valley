import requests
import scrapy
from scrapy.http import HtmlResponse
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import time
from scraper.folksongs.folksongs.items import FolksongsArchiveItem


class FolkSongArchiveSpider(scrapy.Spider):
    name = "folksongs::Archive"
    allowed_domains = ["songbat.com"]
    start_urls = [
        'http://songbat.com/archive/'
    ]

    # default callback func
    # must return Iterable[Request] / None / Item
    def parse(self, response: HtmlResponse):
        regions = response.xpath("/html/body//h3")
        for region_header in regions:
            region = BeautifulSoup(region_header.extract()).get_text()
            print(region)
            x = region_header.xpath("./following-sibling::ul[1]//li")

            yield from response.follow_all(
                region_header.xpath("./following-sibling::ul[1]//li//@href"),
                callback=self.parse_lyric,
                cb_kwargs={"region": region}
            )
            # for song in region_header.xpath("./following-sibling::ul[1]//li"):
            #     song = BeautifulSoup(song.extract())
            #     yield
            #     print(song["href"])


    def parse_lyric(self, response: HtmlResponse, region: str):
        title = BeautifulSoup(response.xpath("/html/body//h1[@class='song-title']").extract_first()).get_text()
        lyric = BeautifulSoup(response.xpath("/html/body//div[@class='song-lyrics']").extract_first()).get_text()
        lyric_eng = BeautifulSoup(response.xpath("/html/body//div[@class='song-lyrics-english']").extract_first()).get_text()
        yield FolksongsArchiveItem(
            region=region,
            title=title,
            lyric=lyric,
            lyric_eng=lyric_eng
        )

