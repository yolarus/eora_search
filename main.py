import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from langchain_core.documents import Document
from chromadb.config import Settings
from langchain_gigachat.embeddings.gigachat import GigaChatEmbeddings
from langchain_chroma import Chroma
from os import getenv
from langchain.text_splitter import CharacterTextSplitter

from dotenv import load_dotenv
from rich.console import Console
from rich.style import Style
from rich.text import Text
load_dotenv('.env')

ACCESS_KEY = getenv('ACCESS_KEY')

console = Console()
ua = UserAgent()

urls = [
    'https://eora.ru/cases/promyshlennaya-bezopasnost',
    'https://eora.ru/cases/lamoda-systema-segmentacii-i-poiska-po-pohozhey-odezhde',
    'https://eora.ru/cases/navyki-dlya-golosovyh-assistentov/karas-golosovoy-assistent',
    'https://eora.ru/cases/assistenty-dlya-gorodov',
    'https://eora.ru/cases/avtomatizaciya-v-promyshlennosti/chemrar-raspoznovanie-molekul',
    'https://eora.ru/cases/zeptolab-skazki-pro-amnyama-dlya-sberbox',
    'https://eora.ru/cases/goosegaming-algoritm-dlya-ocenki-igrokov',
    'https://eora.ru/cases/dodo-pizza-robot-analitik-otzyvov', 'https://eora.ru/cases/ifarm-nejroset-dlya-ferm',
    'https://eora.ru/cases/zhivibezstraha-navyk-dlya-proverki-rodinok',
    'https://eora.ru/cases/sportrecs-nejroset-operator-sportivnyh-translyacij',
    'https://eora.ru/cases/avon-chat-bot-dlya-zhenshchin',
    'https://eora.ru/cases/navyki-dlya-golosovyh-assistentov/navyk-dlya-proverki-loterejnyh-biletov',
    'https://eora.ru/cases/computer-vision/iss-analiz-foto-avtomobilej', 'https://eora.ru/cases/purina-master-bot',
    'https://eora.ru/cases/skinclub-algoritm-dlya-ocenki-veroyatnostej',
    'https://eora.ru/cases/skolkovo-chat-bot-dlya-startapov-i-investorov',
    'https://eora.ru/cases/purina-podbor-korma-dlya-sobaki', 'https://eora.ru/cases/purina-navyk-viktorina',
    'https://eora.ru/cases/dodo-pizza-pilot-po-avtomatizacii-kontakt-centra',
    'https://eora.ru/cases/dodo-pizza-avtomatizaciya-kontakt-centra',
    'https://eora.ru/cases/icl-bot-sufler-dlya-kontakt-centra',
    'https://eora.ru/cases/s7-navyk-dlya-podbora-aviabiletov', 'https://eora.ru/cases/workeat-whatsapp-bot',
    'https://eora.ru/cases/absolyut-strahovanie-navyk-dlya-raschyota-strahovki',
    'https://eora.ru/cases/kazanexpress-poisk-tovarov-po-foto',
    'https://eora.ru/cases/kazanexpress-sistema-rekomendacij-na-sajte',
    'https://eora.ru/cases/intels-proverka-logotipa-na-plagiat',
    'https://eora.ru/cases/karcher-viktorina-s-voprosami-pro-uborku',
    'https://eora.ru/cases/chat-boty/purina-friskies-chat-bot-na-sajte',
    'https://eora.ru/cases/nejroset-segmentaciya-video',
    'https://eora.ru/cases/chat-boty/essa-nejroset-dlya-generacii-rolikov',
    'https://eora.ru/cases/qiwi-poisk-anomalij',
    'https://eora.ru/cases/frisbi-nejroset-dlya-raspoznavaniya-pokazanij-schetchikov',
    'https://eora.ru/cases/skazki-dlya-gugl-assistenta',
    'https://eora.ru/cases/chat-boty/hr-bot-dlya-magnit-kotoriy-priglashaet-na-sobesedovanie'
]


def search(user_question: str) -> list:
    headers = {'User-Agent': ua.random}
    documents = []
    for url in urls:
        result = []
        response = requests.get(url, headers=headers)
        response.encoding = 'utf-8'
        html_content = response.text
        soup = BeautifulSoup(html_content, 'html.parser')
        divs = soup.find_all('div')
        divs = [div for div in divs if
                div.get('class') is not None and any([item.endswith('__artboard') for item in div.get('class')])]
        for div in divs[10:len(divs)-3]:
            result.append(' '.join(div.text.split()))
        name = ' '.join(divs[10].text.split())
        text = '. '.join(result).replace('\xa0', '').replace('/', '').replace('\n', ' ')[:1500]
        documents.append(Document(page_content=text, metadata={'source': url, 'name': name}))
    splitter = CharacterTextSplitter(separator="\n\n", chunk_size=500, chunk_overlap=50)
    chunks = []
    for doc_ in documents:
        chunks.extend(splitter.create_documents([doc_.page_content], [doc_.metadata]))
    embeddings = GigaChatEmbeddings(
        credentials=ACCESS_KEY, verify_ssl_certs=False
    )
    db = Chroma.from_documents(
        chunks,
        embeddings,
        client_settings=Settings(anonymized_telemetry=False),
    )
    question_ = (f"Запрос: Дан вопрос от пользователя ({user_question}), "
                 f"по которому необходимо найти наиболее подходящие документы, "
                 f"чтобы показать ему наиболее близкие кейсы, которые решила компания.\n"
                 f"Найти документы, максимально соответствующие запросу.")
    docs_ = db.similarity_search(question_, k=2)
    return docs_


if __name__ == '__main__':
    question = input('Какой у вас вопрос?: ')
    docs = search(question)
    print('Мы нашли следующие кейсы, наиболее подходящие под Ваш запрос:')
    for i, doc in enumerate(docs, start=1):
        name = ""
        for j, word in enumerate(doc.metadata["name"].split()):
            name += f'{word} '
            if len(name) >= 50:
                if j < len(doc.metadata["name"].split()) - 1:
                    name = f'{name[:-1]}...'
                break
        text = Text(f'{i}. {name}', style=Style(link=doc.metadata["source"]))
        console.print(text)
        'Что вы можете сделать для ритейлера?'
