import os
from flask import Flask, render_template, request,send_from_directory
import joblib
import numpy as np
import pandas as pd
import re
import string
import re
import pyarabic.araby as araby


regex_url_step1 = r'(?=http)[^\s]+'
regex_url_step2 = r'(?=www)[^\s]+'
regex_url = r'(http(s)?:\/\/.)?(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)'
regex_mention = r'@[\w\d]+'
regex_email = r'\S+@\S+'
redundant_punct_pattern = r'([!\"#\$%\'\(\)\*\+,\.:;\-<=·>?@\[\\\]\^_ـ`{\|}~—٪’،؟`୍“؛”ۚ【»؛\s+«–…‘]{2,})'

COMMA = u'\u060C'
SEMICOLON = u'\u061B'
QUESTION = u'\u061F'
HAMZA = u'\u0621'
ALEF_MADDA = u'\u0622'
ALEF_HAMZA_ABOVE = u'\u0623'
WAW_HAMZA = u'\u0624'
ALEF_HAMZA_BELOW = u'\u0625'
YEH_HAMZA = u'\u0626'
ALEF = u'\u0627'
BEH = u'\u0628'
TEH_MARBUTA = u'\u0629'
TEH = u'\u062a'
THEH = u'\u062b'
JEEM = u'\u062c'
HAH = u'\u062d'
KHAH = u'\u062e'
DAL = u'\u062f'
THAL = u'\u0630'
REH = u'\u0631'
ZAIN = u'\u0632'
SEEN = u'\u0633'
SHEEN = u'\u0634'
SAD = u'\u0635'
DAD = u'\u0636'
TAH = u'\u0637'
ZAH = u'\u0638'
AIN = u'\u0639'
GHAIN = u'\u063a'
TATWEEL = u'\u0640'
FEH = u'\u0641'
QAF = u'\u0642'
KAF = u'\u0643'
LAM = u'\u0644'
MEEM = u'\u0645'
NOON = u'\u0646'
HEH = u'\u0647'
WAW = u'\u0648'
ALEF_MAKSURA = u'\u0649'
YEH = u'\u064a'
MADDA_ABOVE = u'\u0653'
HAMZA_ABOVE = u'\u0654'
HAMZA_BELOW = u'\u0655'
ZERO = u'\u0660'
ONE = u'\u0661'
TWO = u'\u0662'
THREE = u'\u0663'
FOUR = u'\u0664'
FIVE = u'\u0665'
SIX = u'\u0666'
SEVEN = u'\u0667'
EIGHT = u'\u0668'
NINE = u'\u0669'
PERCENT = u'\u066a'
DECIMAL = u'\u066b'
THOUSANDS = u'\u066c'
STAR = u'\u066d'
MINI_ALEF = u'\u0670'
ALEF_WASLA = u'\u0671'
FULL_STOP = u'\u06d4'
BYTE_ORDER_MARK = u'\ufeff'

# Diacritics
FATHATAN = u'\u064b'
DAMMATAN = u'\u064c'
KASRATAN = u'\u064d'
FATHA = u'\u064e'
DAMMA = u'\u064f'
KASRA = u'\u0650'
SHADDA = u'\u0651'
SUKUN = u'\u0652'

#Ligatures
LAM_ALEF = u'\ufefb'
LAM_ALEF_HAMZA_ABOVE = u'\ufef7'
LAM_ALEF_HAMZA_BELOW = u'\ufef9'
LAM_ALEF_MADDA_ABOVE = u'\ufef5'
SIMPLE_LAM_ALEF = u'\u0644\u0627'
SIMPLE_LAM_ALEF_HAMZA_ABOVE = u'\u0644\u0623'
SIMPLE_LAM_ALEF_HAMZA_BELOW = u'\u0644\u0625'
SIMPLE_LAM_ALEF_MADDA_ABOVE = u'\u0644\u0622'


HARAKAT_PAT = re.compile(u"["+u"".join([FATHATAN, DAMMATAN, KASRATAN,
                                        FATHA, DAMMA, KASRA, SUKUN,
                                        SHADDA])+u"]")

HAMZAT_PAT = re.compile(u"["+u"".join([WAW_HAMZA, YEH_HAMZA])+u"]")

ALEFAT_PAT = re.compile(u"["+u"".join([ALEF_MADDA, ALEF_HAMZA_ABOVE,
                                       ALEF_HAMZA_BELOW, HAMZA_ABOVE,
                                       HAMZA_BELOW])+u"]")
LAMALEFAT_PAT = re.compile(u"["+u"".join([LAM_ALEF,
                                          LAM_ALEF_HAMZA_ABOVE,
                                          LAM_ALEF_HAMZA_BELOW,
                                          LAM_ALEF_MADDA_ABOVE])+u"]")


""" https://github.com/cltk/cltk/blob/master/cltk/corpus/arabic/alphabet.py """
WESTERN_ARABIC_NUMERALS = ['0','1','2','3','4','5','6','7','8','9']

#EASTERN_ARABIC_NUMERALS = [u'\u06F0', u'\u06F1', u'\u06F2', u'\u06F3', u'\u0664', u'\u06F5', u'\u0666', u'\u06F7', u'\u06F8', u'\u06F9']
EASTERN_ARABIC_NUMERALS = [u'۰', u'۱', u'۲', u'۳', u'٤', u'۵', u'٦', u'۷', u'۸', u'۹']

prefix_list = ["ال", "و", "ف", "ب", "ك", "ل", "لل", "\u0627\u0644", "\u0648", "\u0641", "\u0628", "\u0643", "\u0644", "\u0644\u0644", "س"]
suffix_list = ["ه", "ها", "ك", "ي", "هما", "كما", "نا", "كم", "هم", "هن", "كن",
                 "ا", "ان", "ين", "ون", "وا", "ات", "ت", "ن", "ة",
                "\u0647", "\u0647\u0627", "\u0643", "\u064a", "\u0647\u0645\u0627", "\u0643\u0645\u0627", "\u0646\u0627", "\u0643\u0645", "\u0647\u0645", "\u0647\u0646", "\u0643\u0646",
                "\u0627", "\u0627\u0646", "\u064a\u0646", "\u0648\u0646", "\u0648\u0627", "\u0627\u062a", "\u062a", "\u0646", "\u0629" ]

# the never_split list is ussed with the transformers library
prefix_symbols = [ x+"+" for x in prefix_list]
suffix_symblos = [ "+"+x for x in suffix_list]
never_split_tokens = list(set(prefix_symbols+suffix_symblos))

eastern_to_western_numerals = {}
for i in range(len(EASTERN_ARABIC_NUMERALS)):
    eastern_to_western_numerals[EASTERN_ARABIC_NUMERALS[i]] = WESTERN_ARABIC_NUMERALS[i]

# Punctuation marks
COMMA = u'\u060C'
SEMICOLON = u'\u061B'
QUESTION = u'\u061F'

# Other symbols
PERCENT = u'\u066a'
DECIMAL = u'\u066b'
THOUSANDS = u'\u066c'
STAR = u'\u066d'
FULL_STOP = u'\u06d4'
MULITIPLICATION_SIGN = u'\u00D7'
DIVISION_SIGN = u'\u00F7'

arabic_punctuations = COMMA + SEMICOLON + QUESTION + PERCENT + DECIMAL + THOUSANDS + STAR + FULL_STOP + MULITIPLICATION_SIGN + DIVISION_SIGN
all_punctuations = string.punctuation + arabic_punctuations + '()[]{}'

all_punctuations = ''.join(list(set(all_punctuations)))


def strip_tashkeel(text):
    text = HARAKAT_PAT.sub('', text)
    text = re.sub(u"[\u064E]", "", text,  flags=re.UNICODE) # fattha
    text = re.sub(u"[\u0671]", "", text,  flags=re.UNICODE) # waSla
    return text 


def strip_tatweel(text):
    return re.sub(u'[%s]' % TATWEEL, '', text)


def remove_non_arabic(text):
    return ' '.join(re.sub(u"[^\u0621-\u063A\u0640-\u0652 ]", " ", text,  flags=re.UNICODE).split())


def keep_arabic_english_n_symbols(text):
    return ' '.join(re.sub(u"[^\u0621-\u063A\u0640-\u064aa-zA-Z#@_:/ ]", "", text,  flags=re.UNICODE).split())


def normalize_hamza(text):
    text = ALEFAT_PAT.sub(ALEF, text)
    return HAMZAT_PAT.sub(HAMZA, text)


def normalize_spellerrors(text):
    text = re.sub(u'[%s]' % TEH_MARBUTA, HEH, text)
    return re.sub(u'[%s]' % ALEF_MAKSURA, YEH, text)


def normalize_lamalef(text):
    return LAMALEFAT_PAT.sub(u'%s%s'%(LAM, ALEF), text)


def normalize_arabic_text(text):
    text = remove_non_arabic(text)
    text = strip_tashkeel(text)
    text = strip_tatweel(text)
    text = normalize_lamalef(text)
    text = normalize_hamza(text)
    text = normalize_spellerrors(text)
    return text


def remove_underscore(text):
    return ' '.join(text.split('_'))


def remove_retweet_tag(text):
    return re.compile('\#').sub('', re.compile('rt @[a-zA-Z0-9_]+:|@[a-zA-Z0-9_]+').sub('', text).strip())


def replace_emails(text):
    emails = re.findall(r'[\w\.-]+@[\w\.-]+', text)
    for email in emails:
        text = text.replace(email,' يوجدايميل ')
        #text = text.replace(email,' hasEmailAddress ')
    return text

def replace_urls(text):
    return re.sub(r"http\S+|www.\S+", " يوجدرابط ", text)
    #return re.sub(r"http\S+|www.\S+", " hasURL ", text)

def convert_eastern_to_western_numerals(text):
    for num in EASTERN_ARABIC_NUMERALS:
        text = text.replace(num, eastern_to_western_numerals[num])
    return text

def remove_all_punctuations(text):
    for punctuation in all_punctuations:
        text = text.replace(punctuation, ' ')
    return text

def remove_emojis(text):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def replace_phone_numbers(text):
    return re.sub(r'\d{10}', ' يوجدرقمهاتف ', text)
    # return re.sub(r'\d{10}', ' hasPhoneNumber ', text)

def remove_extra_spaces(text):
    return ' '.join(text.split())

'''
very important note:
    The order of the execution of the these function is extremely crucial.
'''
def normalize_tweet(text):
    new_text = text.lower()
    new_text = normalize_hamza(new_text)
    new_text = strip_tashkeel(new_text)
    new_text = strip_tatweel(new_text)
    new_text = normalize_lamalef(new_text)
    new_text = normalize_spellerrors(new_text)
    new_text = remove_retweet_tag(new_text)
    new_text = replace_emails(new_text)
    new_text = remove_underscore(new_text)
    new_text = replace_phone_numbers(new_text)
    new_text = remove_all_punctuations(new_text)
    new_text = replace_urls(new_text)
    new_text = convert_eastern_to_western_numerals(new_text)
#    new_text = keep_arabic_english_n_symbols(new_text)
    new_text = remove_non_arabic(new_text)
    new_text = remove_extra_spaces(new_text)
    
    return new_text

##############################################################################################################

def remove_elongation(word):
	"""
    :param word:  the input word to remove elongation
    :return: delongated word
    """
	regex_tatweel = r'(\w)\1{2,}'
	# loop over the number of times the regex matched the word
	for index_ in range(len(re.findall(regex_tatweel, word))):
		if re.search(regex_tatweel, word):
			elongation_found = re.search(regex_tatweel, word)
			elongation_replacement = elongation_found.group()[0]
			elongation_pattern = elongation_found.group()
			word = re.sub(elongation_pattern, elongation_replacement, word, flags=re.MULTILINE)
		else:
			break
	return word


def tokenize_arabic_words_farasa(line_input, farasa):
    segmented_line=[]
    line_farasa = farasa.segmentLine(line_input)
    for index , word in enumerate(line_farasa):
        if word in ['[',']']:
            continue
        if word in ['رابط','بريد','مستخدم'] and line_farasa[index-1] in ['[',']']:
            segmented_line.append('['+word+']')
            continue
        segmented_word=[]
        for token in word.split('+'):
            if token in prefix_list:
                segmented_word.append(token+'+')
            elif token in suffix_list:
                segmented_word.append('+'+token)
            else:
                segmented_word.append(token)
        segmented_line.extend(segmented_word)

    return ' '.join(segmented_line)


def remove_redundant_punct(text):
	text_ = text
	result = re.search(redundant_punct_pattern, text)
	dif = 0
	while result:
		sub = result.group()
		sub = sorted(set(sub), key=sub.index)
		sub = ' ' + ''.join(list(sub)) + ' '
		text = ''.join((text[:result.span()[0]+dif], sub, text[result.span()[1]+dif:]))
		text_ = ''.join((text_[:result.span()[0]], text_[result.span()[1]:])).strip()
		dif = abs(len(text) - len(text_))
		result = re.search(redundant_punct_pattern, text_)
	text = re.sub(r'\s+', ' ', text)
	return text.strip()


def preprocess(text, do_farasa_tokenization=True , farasa=None):
		"""
		Preprocess takes an input text line an applies the same preprocessing used in araBERT 
		pretraining
		Args:
		text (string): inout text string
		farasa (JavaGateway): pass a py4j gateway to the FarasaSegmenter.jar file 
		Example: 
		from py4j.java_gateway import JavaGateway
		gateway = JavaGateway.launch_gateway(classpath='./FarasaSegmenterJar.jar')
		farasa = gateway.jvm.com.qcri.farasa.segmenter.Farasa()
		processed_text = preprocess("Some_Text",do_farasa_tokenization=True , farasa=farasa)

		"""
		text=str(text)
		processing_tweet = araby.strip_tashkeel(text)
		processing_tweet = re.sub(r'\d+\/[ء-ي]+\/\d+\]', '', processing_tweet)
		processing_tweet = re.sub('ـ', '', processing_tweet)	
		processing_tweet = re.sub('[«»]', ' " ', processing_tweet)
		#replace the [رابط] token with space if you want to clean links
		#processing_tweet = re.sub(regex_url_step1, '[رابط]', processing_tweet)
		processing_tweet = re.sub(regex_url_step1, ' ', processing_tweet)
		#processing_tweet = re.sub(regex_url_step2, '[رابط]', processing_tweet)
		processing_tweet = re.sub(regex_url_step2, ' ', processing_tweet)
		#processing_tweet = re.sub(regex_url, '[رابط]', processing_tweet)
		processing_tweet = re.sub(regex_url, ' ', processing_tweet)
		#processing_tweet = re.sub(regex_email, '[بريد]', processing_tweet)
		processing_tweet = re.sub(regex_email, ' ', processing_tweet)
		#processing_tweet = re.sub(regex_mention, '[مستخدم]', processing_tweet)
		processing_tweet = re.sub(regex_mention, ' ', processing_tweet)
		processing_tweet = re.sub('…', r'\.', processing_tweet).strip()
		processing_tweet = remove_redundant_punct(processing_tweet)

		processing_tweet = re.sub(r'\[ رابط \]|\[ رابط\]|\[رابط \]', ' ', processing_tweet)
		processing_tweet = re.sub(r'\[ بريد \]|\[ بريد\]|\[بريد \]', ' ', processing_tweet)
		processing_tweet = re.sub(r"\[ مستخدم \]|\[ مستخدم\]|\[مستخدم \]|\[' مستخدم ]", ' ', processing_tweet)

		processing_tweet = remove_elongation(processing_tweet)
		if do_farasa_tokenization and farasa is not None:
			processing_tweet = tokenize_arabic_words_farasa(processing_tweet, farasa)
		return processing_tweet.strip()

#################################################################################################################
def tweet_preprcessing(tweet):
	text_preprocessed = preprocess(tweet, do_farasa_tokenization=True)
	preprocessed_tweet= normalize_arabic_text(text_preprocessed)
	return preprocessed_tweet
#################################################################################################################
# __name__ is equal to app.py
app = Flask(__name__)

# load model from model.pck
model = joblib.load('LinearSVC_model.pkl')

@app.route('/favicon.ico') 
def favicon(): 
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predict", methods=["GET","POST"])
def predict():
	keys_dictionary   = {0:"Egypt",1:"Palestine",2:"Kuwait",3:"Libya",4:"Qatar",5:"Jordan",6:"Lebanon",7:"Saudi Arabia",
						8:"UAE",9:"Bahrain",10:"Oman",11:"Syria",12:"Algeria",13:"Iraq",14:"Sudan",15:"Morocco",
						16:"Yemen",17:"Tunisia"}

	text =  [request.form['tweet']]
	df = pd.DataFrame({'T':text})

	df['T'] = df['T'].apply(tweet_preprcessing)

	dialect = keys_dictionary.get(model.predict(df['T'])[0])
	
	return render_template("index.html", dialect=dialect)	


if __name__ == "__main__":
    app.run(host='0.0.0.0',port=5000)
