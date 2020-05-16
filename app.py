import streamlit as st
import pandas as pd
import re
import matplotlib.pyplot as plt
import altair as alt

import nltk
from collections import Counter
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from nltk.util import ngrams

def main():
    st.title("**Text Analyzer v0.1**")
    app_mode = st.sidebar.selectbox('Navegation',['Bem Vindos','Word Cloud','Word Counter'])
    if app_mode == 'Bem Vindos':
        st.markdown("***")
        st.markdown(
            "Projeto de desenvolvido durante a aceleração de Data Science da codenation.\n\n"
            "Para mais infocacoes sobre o projeto, segue o link do [GitHub](https://github.com/danvr/streamlit-text-analyzer).")
        st.video("https://www.youtube.com/watch?v=WQ2isQoHMR0")
        st.markdown("***")
        st.markdown("## **Motivação**")
        st.markdown(
    	"O objetivo e resolver uma dor real que envolva análise de dados.\n\n"
        "Durante a fase de validação de um produto, e muito comum que sejam feitas pesquisas "\
        "de usuários para entender as verdadeiras dores. "\
        "O resultado desse processo normalmente gera uma massa de dados textuais, "\
        "onde o profissional de UX normalmente faz um processo manual de contagem de palavras e expressões.\n\n"
        "Essa fermenta se propõem a agilizar esse processo automatizando a contagem de palavras"\
        "de nuvem de palavras e contadores de palavras.\n\n"
        "Espero que seja útil mesmo estando em uma versão muito simplista, mas o foco é resolver o problema, feedbacks são bem vindos para continuar a melhorar"\
        "a ferramenta\n\n"
        "**Boa análise para todos!!**")
        st.markdown("***")
        st.markdown("## *Sobre o Autor*")
        st.markdown("## **Daniel Vieira, Cientista de Dados**\n\n"
        "*Fazendo o mundo melhor resolvendo problemas com dados.*\n\n"
        "* [Linkedin](https://www.linkedin.com/in/danielvieiraroberto/)\n\n"
        "* [Git Hub](https://github.com/danvr)")
      
    elif app_mode == 'Word Cloud':
        st.markdown("***")
        st.markdown("# Word Cloud")
        st.markdown(
            "Gerador interativo de Word Cloud(Nuvem de Plavras) onde o tamanho da palavra "\
            "é correspondente a frequência e relevância.\n\n")
        st.markdown("## **Como Usar**")
        st.markdown(
            "* Faça upload de arquivo .csv\n\n"
            "* Escolha a coluna\n\n"
            "* Obtenha insights!")        
        word_cloud_generetor()
    elif app_mode == 'Word Counter':
        st.markdown("***")
        st.markdown("# Word Counter")
        st.markdown("Contador de palavras parametrizado por número de n-gramas"\
        "(sequência continua de deternimado número itens ou palavras)")
        st.markdown("## **Como Usar**")
        st.markdown(
            "* Faça upload de arquivo .csv\n\n"
            "* Escolha a coluna\n\n"
            "* Obtenha insights!")    
        word_counter()


def word_cloud_generetor():
    uploaded_file = st.file_uploader('Choose a CSV file', type="csv")
    if uploaded_file is not None:
        df = load_file(uploaded_file)
        df_colmuns = df.select_dtypes(include=['object']).columns
        seleted_colmun = st.sidebar.selectbox('Select Colmun to display Word Cloud',df_colmuns)
        max_words_filter = st.sidebar.slider('Max Words', 5, 100, 10,1)
        relative_scaling_filter = st.sidebar.slider('Scaling', 0.0, 1.0, 0.5,0.01)
        plot_words(df[seleted_colmun], 'portuguese', max_words_filter, relative_scaling_filter)
       


@st.cache(allow_output_mutation=True)
def load_file(uploaded_file):

    df = pd.read_csv(uploaded_file)

    def validate_numeric(column):
        try:
            pd.to_numeric(df[column])
        except:
            return False
        return True

    def validate_datetime(column):
        try:
                pd.to_datetime(df[column])
        except:
            return False
        return True

    for column in df.columns:
        if validate_numeric(column) == True:
            df[column] =  pd.to_numeric(df[column])
        elif validate_datetime(column) == True:
            df[column] = pd.to_datetime(df[column])
        else:
            pass

    return df


def plot_words(pandas_series,language,max_words,relative_scaling):
    stopwords = set(nltk.corpus.stopwords.words(language))
    word_list = create_word_list(pandas_series,stopwords)

    wordcloud = create_word_cloud(stopwords,word_list,max_words,relative_scaling)
                                       
    plt.figure(figsize=(15,10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
  
    st.pyplot()
   
    data  = pd.Series(wordcloud.words_)
    st.table(data)

      
def create_word_list(pandas_series,stopwords):
    word_list = pandas_series.fillna("",inplace = True)
    word_list = " ".join(word for word in pandas_series)
    word_list = word_list.lower()
    word_list = re.sub(r'[-./?!,":;()\']',' ', word_list)
    word_list = [word for word in word_list.lower().split() if not word in stopwords]
    word_list = (" ".join(word_list))
    return word_list


def create_word_cloud(stopwords,word_list,max_words,relative_scaling):
    wordcloud = WordCloud(
        width=800,
        height=600,
        background_color="white",
        stopwords=stopwords,
        max_words=max_words,
        relative_scaling = relative_scaling,
        collocations = True
    ).generate(word_list)  
    return wordcloud


def word_counter():
    uploaded_file = st.file_uploader('Choose a CSV file', type="csv")
    if uploaded_file is not None:
        df = load_file(uploaded_file)
        seleted_colmun = st.sidebar.selectbox('Select Colmun count the words',df.select_dtypes(include=['object']).columns)
        ngrams_filter = st.sidebar.number_input('ngram', 1, 10, 2,1)
        stopwords = set(nltk.corpus.stopwords.words('portuguese'))
        word_list = create_word_list(df[seleted_colmun], stopwords)
        ngram = extract_ngrams(word_list,ngrams_filter)
                                
        counter = Counter(ngram)
        word_count = pd.DataFrame(
            counter.most_common(5),
            columns=['ngram', 'Freq']).set_index('ngram')
        word_count['Freq'] =  (word_count['Freq']/sum(counter.values()))*100
        st.write(word_count)
        return word_count 


def extract_ngrams(data, num):
    n_grams = ngrams(nltk.word_tokenize(data), num)
    return [ ' '.join(grams) for grams in n_grams]


if __name__=="__main__":
    nltk.download('stopwords')
    nltk.download('punkt')
    main()







