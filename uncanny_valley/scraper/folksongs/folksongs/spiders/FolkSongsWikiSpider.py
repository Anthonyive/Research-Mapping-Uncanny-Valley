import requests
import scrapy
from scrapy.http import HtmlResponse
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import time
from scraper.folksongs.folksongs.items import FolksongsItem


class FolkSongWikiSpider(scrapy.Spider):
    name = "folksongs::wikipedia"
    allowed_domains = ["lyrics.com"]
    start_urls = [
        'https://en.wikipedia.org/wiki/List_of_folk_songs_by_Roud_number?oldformat=true',
    ]

    # default callback func
    # must return Iterable[Request] / None / Item
    def parse(self, response: HtmlResponse):

        # list of songs span from 5 - 40
        for table_num in range(5, 41):
            div = response.xpath(f"//body[1]/div[3]/div[3]/div[5]/div[1]/div[{table_num}]")
            # table, list of songs in this div
            tables = div.xpath(".//td")
            for table_idx, table in enumerate(tables, 1):
                for idx, song in enumerate(table.xpath(".//li"), 1):
                    song: str = BeautifulSoup(song.extract()).get_text()

                    # div 5's table 1 is special: doesn't have index
                    # manually add
                    if table_num == 5 and table_idx == 1:
                        song = str(idx) + ". " + song

                    print(song)
                    # can be multiple ".", but only split 1 to separate idx & rest
                    song_id, song_title = song.split(".", 1)

                    if song_title != "No record":
                        first_quote = song_title.find('"')
                        second_quote = song_title.find('"', first_quote + 1)
                        song_title = song_title[first_quote + 1:second_quote]
                    print("\t", song_id, song_title)

                    lyrics_url = urljoin("https://www.lyrics.com/lyrics/", song_title)

                    # time.sleep(5)
                    yield response.follow(lyrics_url,
                                          callback=self.parse_lyrics_com,
                                          cb_kwargs={
                                              "songid": song_id,
                                              "title": song_title
                                          })

    def parse_lyrics_com(self, response: HtmlResponse, songid: str, title: str):
        lyric = ""
        try:
            lyric_section = response.css("div.sec-lyric p.lyric-meta-title")
            # ignore title doesn't match
            lyric_title = BeautifulSoup(lyric_section.extract_first()).get_text()
            if lyric_title == title:
                lyric_href = lyric_section.css("a::attr(href)").extract_first()
                request = requests.get(urljoin("https://www.lyrics.com/lyrics/", lyric_href))
                resp = scrapy.Selector(request)
                lyric = resp.xpath("//pre[@id='lyric-body-text']")
                lyric = BeautifulSoup(lyric.extract_first()).get_text()
        except:
            pass

        yield FolksongsItem(
            songid=songid,
            title=title,
            lyric=lyric
        )

