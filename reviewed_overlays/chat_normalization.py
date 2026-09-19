import re

CORRECTIONS = {
    "dont": "don't", "im": "I'm", "thats": "that's", "doesnt": "doesn't",
    "didnt": "didn't", "ive": "I've", "isnt": "isn't", "theres": "there's",
    "havent": "haven't", "wouldnt": "wouldn't", "wasnt": "wasn't",
    "shouldnt": "shouldn't", "youre": "you're", "alot": "a lot",
    "nevermind": "never mind", "okey": "okay", "noone": "no one",
    "that'll": "that will", "arent": "aren't", "tryed": "tried",
    "hai": "hi", "fiesty": "feisty", "prolly": "probably", "wierd": "weird",
    "jdk": "idk", "couldnt": "couldn't", "somthing": "something",
    "taht": "that", "waht": "what", "helpme": "help me", "wich": "which",
    "thnks": "thanks", "allright": "all right", "alright": "all right",
    "damnit": "damn it", "dosent": "doesn't", "aswell": "as well",
    "oki": "okay", "belive": "believe", "danke": "thanks",
    "behaviour": "behavior", "doesn": "doesn't", "happend": "happened",
    "allready": "already", "ohmy": "oh my", "theyre": "they're",
    "colour": "color", "kool": "cool", "wats": "what's", "wat": "what",
    "envyng": "envying", "wha": "what", "ello": "hello", "kernal": "kernel",
    "thansk": "thanks", "ooops": "oops", "awsome": "awesome", "wut": "what",
    "dat": "that", "halp": "help", "hellow": "hello", "hellp": "help",
    "youll": "you will", "didn": "didn't", "reccomend": "recommend",
    "problemo": "problem", "tryin": "trying", "diffrent": "different",
    "liek": "like", "allways": "always", "somethin": "something",
    "guyz": "guys", "formated": "formatted", "thank's": "thanks",
    "lookin": "looking", "avg": "average", "thaks": "thanks",
    "thr": "the", "nothin": "nothing", "how'd": "how did",
    "automaticly": "automatically", "programm": "program", "rox": "rocks",
    "yessir": "yes sir", "shure": "sure", "abt": "about",
    "cens0red": "censored", "n00dle": "noodle", "h0nestly": "honestly",
    "nic3": "nice", "passw0rd": "password", "please2help": "please help",
    "trollin": "trolling", "whatnow": "what now", "ubuntuguy": "ubuntu guy",
    "updtaes": "updates", "personnal": "personal", "descompressed": "decompressed",
    "usbview": "usb view", "pckgs": "packages", "nevr": "never", "kthx": "k thx",
    "onlt": "only", "knowedge": "knowledge", "where'd": "where did",
    "forgivven": "forgiven", "hten": "then", "sreenshot": "screenshot",
    "messager": "messenger", "aray": "array", "wirlesly": "wirelessly",
    "risksy": "risky", "thetruth": "the truth", "navtive": "native", "titun": "titan",
    "actualy": "actually", "permmisions": "permissions", "infact": "in fact",
    "wiseass": "wise ass", "adobeairinsstaller": "adobe air installer",
    "avaible": "available", "widthxheight": "width x height", "mionth": "month",
    "lanugage": "language", "helkp": "help", "minimalistic": "minimal",
    "problemes": "problems", "shoult": "should", "doyetmorestuff": "do yet more stuff",
    "uptodate": "up to date", "disapeared": "disappeared", "blastedt": "blasted",
    "kaffiene": "caffeine", "attetion": "attention", "betteer": "better",
    "exta": "extra", "runlevels": "run levels", "flightgear": "flight gear",
    "realy": "really", "directiry": "directory", "mabyyyyy": "maybe", "nvmd": "nvm",
    "brotha": "bro", "thany": "thank", "dowsn't": "doesn't", "pricedrop": "price drop",
    "youv": "you've", "nvidi": "nvidia", "buton": "button",
    "spontaniously": "spontaneously", "ubunto": "ubuntu",
    "allsystemsarego": "all systems are go", "whrong": "wrong",
    "automaticaly": "automatically", "partiion": "partition", "avaiable": "available",
    "syggest": "suggest", "idears": "ideas", "allobjects": "all objects",
    "respritories": "repositories", "nvrm": "nvm", "grammer": "grammar",
    "totlal": "total", "iguess": "i guess", "tthe": "the", "hasnt": "hasn't",
    "runit": "run it", "tsee": "see", "gowith": "go with", "pingpong": "ping pong",
    "alreay": "already", "corrction": "correction",
    "autopartitionning": "auto partitioning", "develpers": "developers",
    "hopfully": "hopefully", "dono": "don't know", "dunno": "don't know",
    "camra": "camera", "anyfoo": "anyone", "oretty": "pretty", "exactley": "exactly",
    "safemode": "safe mode", "learnings": "learning",
    "randomnickname": "random nickname", "cheatcode": "cheat code", "algo": "algorithm",
    "algos": "algorithms", "have't": "haven't", "anways": "anyways", "nows": "now",
    "cours": "course", "desappears": "disappears", "azuzuers": "azureus",
    "cooly": "cool", "shouldnve": "shouldn't have", "antbody": "anybody",
    "intreptid": "intrepid", "instaleld": "installed", "aand": "and",
    "answere": "answer", "overby": "over by", "seriouslt": "seriously",
    "korganizer": "organizer", "appologies": "apologies", "enyone": "anyone",
    "haayy": "hey", "weirf": "weird", "bewteen": "between",
    "whatyouwant": "what you want", "knida": "kinda", "keybord": "keyboard",
    "perminantly": "permanently", "paswd": "password", "nooone": "no one",
    "desktopsettings": "desktop settings", "relaeased": "released",
    "sellling": "selling", "oanother": "another", "preaction": "reaction",
    "wroung": "wrong", "apprpriately": "appropriately", "happinging": "happening",
    "familar": "familiar", "sparow": "sparrow", "answeres": "answers",
    "iasked": "i asked", "eithre": "either", "somethig": "something",
    "quesions": "questions", "brogram": "program", "definately": "definitely",
    "happyface": "happy face", "exract": "extract", "reserach": "research",
    "thamks": "thanks", "relevan": "relevant", "hokay": "okay", "i'ts": "it's",
    "disregaurd": "disregard", "begginner": "beginner", "qustions": "questions",
    "mangment": "management", "appearently": "apparently", "supperior": "superior",
    "recommand": "recommend", "shoud": "should", "logitch": "logitech",
    "suprised": "surprised", "channelu": "channel", "sorruy": "sorry",
    "checkgmail": "check gmail", "youself": "yourself", "tahts": "that's",
    "doesnotwork": "does not work", "appologize": "apologize",
    "announcment": "announcement", "neww": "new", "permissoins": "permissions",
    "itused": "it used", "instad": "instead", "beatifull": "beautiful",
    "upodates": "updates", "iwas": "I was", "neevermind": "never mind",
    "copywright": "copyright", "messeges": "messages", "sesson": "session",
    "packaghe": "package", "icant": "I can't", "familly": "family", "hyou": "you",
    "yeha": "yeah", "what'd": "what did", "smothly": "smoothly", "builtin": "built in",
    "fielsystem": "filesystem", "wh4t": "what",
    "alacarte": "a la carte", "orderd": "ordered",
    # --- additions from residual-vocabulary classification pass ---
    'acces': 'access', 'acess': 'access', 'adress': 'address', 'agian': 'again', 
    'ahve': 'have', 'aight': 'alright', 'aint': "ain't", 'any1': 'anyone', 
    'atleast': 'at least', 'bandwith': 'bandwidth', 'basicly': 'basically', 
    'becuase': 'because', 'begining': 'beginning', 'beleive': 'believe', 
    'bittorent': 'bittorrent', 'bla': 'blah', 'borked': 'broken', 'buntu': 'ubuntu', 
    'carefull': 'careful', 'cmon': 'come on', 'colours': 'colors', 'comand': 'command', 
    'comming': 'coming', 'compatability': 'compatibility', 'completly': 'completely', 
    'da': 'the', 'defualt': 'default', 'dependancies': 'dependencies', 
    'dident': "didn't", 'differnet': 'different', 'differnt': 'different', 
    "doens't": "doesn't", 'doenst': "doesn't", 'doesent': "doesn't", 'doin': 'doing', 
    "dosen't": "doesn't", "dosn't": "doesn't", 'dosnt': "doesn't", 'ect': 'etc', 
    'ehm': 'um', 'enviroment': 'environment', 'errr': 'err', 'espanol': 'español', 
    'espaol': 'español', 'every1': 'everyone', 'everytime': 'every time', 'ew': 'eww', 
    'eyecandy': 'eye candy', 'favourite': 'favorite', 'formating': 'formatting', 
    'gettin': 'getting', 'gf': 'girlfriend', 'goin': 'going', 'goodluck': 'good luck', 
    'gues': 'guess', 'happends': 'happens', 'harddrive': 'hard drive', 
    'harddrives': 'hard drives', 'hardrive': 'hard drive', 'hav': 'have', 
    "havn't": "haven't", 'havnt': "haven't", 'helo': 'hello', 'heya': 'hey', 
    'hrm': 'hmm', 'hrmm': 'hmm', 'hrs': 'hours', 'hte': 'the', 'humm': 'hmm', 
    'hve': 'have', 'hwo': 'how', 'iam': 'I am', 'instalation': 'installation', 
    'instaled': 'installed', 'intall': 'install', 'isn': "isn't", 'isntall': 'install', 
    'isntalled': 'installed', 'jsut': 'just', 'jus': 'just', 'kewl': 'cool', 
    'kno': 'know', 'knwo': 'know', 'm8': 'mate', 'maby': 'maybe', 'manualy': 'manually', 
    'ment': 'meant', 'mmmm': 'hmm', "mp3's": 'mp3s', 'naw': 'no', 'nothign': 'nothing', 
    'ofcourse': 'of course', 'oic': 'oh I see', 'okies': 'okay', 'okk': 'ok', 
    'pannel': 'panel', 'parition': 'partition', 'partion': 'partition', 
    'partions': 'partitions', 'partiton': 'partition', 'proberly': 'probably', 
    'proble': 'problem', 'puter': 'computer', 'quesiton': 'question', 
    'realise': 'realize', 'recieve': 'receive', 'recieved': 'received', 
    'recognise': 'recognize', 'recognised': 'recognized', 'recomend': 'recommend', 
    'redownload': 're-download', 'refering': 'referring', 'rightclick': 'right click', 
    'runing': 'running', 'rythmbox': 'rhythmbox', 'sa': 'so', 'sence': 'sense', 
    'seperate': 'separate', 'shoudl': 'should', 'similer': 'similar', 
    'smth': 'something', 'some1': 'someone', 'somethign': 'something', 
    'someting': 'something', 'somone': 'someone', 'soo': 'so', 'sooo': 'so', 
    'sory': 'sorry', 'sould': 'should', 'soz': 'sorry', 'stoped': 'stopped', 
    'sux': 'sucks', 'teh': 'the', 'tha': 'the', 'thankyou': 'thank you', 
    "that'd": 'that would', 'ther': 'there', 'thers': 'theirs', 'thier': 'their', 
    'thks': 'thanks', 'thnaks': 'thanks', 'thnk': 'think', 'thnx': 'thanks', 
    'tht': 'that', 'tis': 'it is', 'tks': 'thanks', 'tnx': 'thanks', 'ubu': 'ubuntu', 
    'ubunt': 'ubuntu', 'ubunut': 'ubuntu', 'ubunutu': 'ubuntu', 'ubutnu': 'ubuntu', 
    'unbuntu': 'ubuntu', 'unfortunatly': 'unfortunately', 'untill': 'until', 
    'usefull': 'useful', 'useing': 'using', 'vids': 'videos', 'w00t': 'woot', 
    'wa': 'was', "wan't": "won't", 'webpages': 'web pages', 'wel': 'well', 
    'wher': 'where', 'whos': "who's", 'wht': 'what', 'winblows': 'windows', 
    'windoze': 'windows', 'wirless': 'wireless', 'wnat': 'want', 'workin': 'working', 
    'yall': 'you all', 'yeap': 'yeah', 'yeh': 'yeah', 'yepp': 'yep', 'yess': 'yes', 
    'youd': "you'd", 'youve': "you've", 'yr': 'your', 'yrs': 'years', 'yuo': 'you',
    # --- batch 3: closing the count>=100 gap in the still-unclassified list ---
    'msgs': 'messages', 'probly': 'probably', 'sth': 'something', 'def': 'definitely',
    'remeber': 'remember', 'stil': 'still', 'talkin': 'talking', 'realised': 'realized',
    'tring': 'trying', 'becouse': 'because', 'helpfull': 'helpful', 'donno': "don't know",
    'progs': 'programs', 'hummm': 'hmm', 'softwares': 'software', 'soooo': 'so',
    # --- batch 4: 50-99 occurrence band ---
    'abotu': 'about', 'accelleration': 'acceleration', 'addy': 'address', 'adn': 'and',
    'alll': 'all', 'allo': 'hello', 'anoying': 'annoying', 'anythign': 'anything',
    'anythin': 'anything', 'anyting': 'anything', 'anywho': 'anyway',
    'aparently': 'apparently', 'aplications': 'applications', 'apologise': 'apologize',
    'arnt': "aren't", 'askin': 'asking', 'conection': 'connection', 'couse': 'cause',
    'crtl': 'ctrl', 'curiousity': 'curiosity', 'cus': 'because', 'damm': 'damn',
    'dangit': 'dang it', 'definatly': 'definitely', 'definitly': 'definitely',
    'diferent': 'different', 'diffrence': 'difference', 'dnt': "don't", 'dotn': "don't",
    'dum': 'dumb', 'duno': "don't know", 'eachother': 'each other', 'enuff': 'enough',
    'esp': 'especially', 'everthing': 'everything', 'everythign': 'everything',
    'exacly': 'exactly', 'extention': 'extension', 'fav': 'favorite', 'fesity': 'feisty',
    'friggin': 'freaking', 'g2g': 'got to go', 'geting': 'getting', 'googleing': 'googling',
    'gunna': 'gonna', 'hangon': 'hang on', 'helllo': 'hello', 'horiz': 'horizontal',
    'hrmmm': 'hmm', 'htat': 'that', 'hullo': 'hello', 'ima': 'I am going to',
    'imma': 'I am going to', 'infos': 'info', 'infront': 'in front', 'iptable': 'iptables',
    'iz': 'is', 'jaja': 'haha', 'jajaja': 'haha', 'jst': 'just', 'knoe': 'know', 'knw': 'know',
    'konw': 'know', 'langauge': 'language', 'lols': 'lol', 'lulz': 'lol', 'mah': 'my',
    'mornin': 'morning', 'nopes': 'nope', 'normaly': 'normally', 'noticable': 'noticeable',
    'nvida': 'nvidia', 'nvidea': 'nvidia', 'occured': 'occurred', 'ohk': 'ok', 'okok': 'ok',
    'oky': 'okay', 'oups': 'oops', 'paritions': 'partitions', 'partitons': 'partitions',
    'pluged': 'plugged', 'powerfull': 'powerful', 'prefered': 'preferred',
    'prefrences': 'preferences', 'probelm': 'problem', 'propably': 'probably',
    'reccomended': 'recommended', 'recomended': 'recommended', 'reconfig': 'reconfigure',
    'rember': 'remember', 'repositorys': 'repositories', 'rhytmbox': 'rhythmbox',
    'righ': 'right', 'rigth': 'right', 'rly': 'really', 'runnin': 'running', 'sayin': 'saying',
    'shouldent': "shouldn't", 'shud': 'should', 'shutup': 'shut up', 'similiar': 'similar',
    'soemthing': 'something', 'somehting': 'something', 'soory': 'sorry', 'srsly': 'seriously',
    'stuf': 'stuff', 'sytem': 'system', 'tehre': 'there', 'thay': 'they',
    'thinkin': 'thinking', 'thre': 'there', 'tnks': 'thanks', 'totaly': 'totally',
    'tought': 'thought', 'tru': 'true', 'uboto': 'ubuntu', 'ubutu': 'ubuntu',
    'unistall': 'uninstall', 'unsecure': 'insecure', 'usualy': 'usually', 'verry': 'very',
    'virtualisation': 'virtualization', 'wana': 'want to', 'waz': 'was', 'werent': "weren't",
    'wfm': 'works for me', 'whant': 'want', 'whatcha': 'what are you', 'whatsup': "what's up",
    'whay': 'why', 'wiht': 'with', 'woudl': 'would', 'wouldent': "wouldn't", 'yupp': 'yup',
    'zomg': 'omg',
    # --- batch 5: 20-49 occurrence band ---
    'absolutly': 'absolutely', 'accesories': 'accessories', 'accessable': 'accessible',
    'accross': 'across', 'acheive': 'achieve', 'acomplish': 'accomplish', 'acount': 'account',
    'actaully': 'actually', 'actuall': 'actual', 'actully': 'actually', 'acutally': 'actually',
    'adaptr': 'adapter', 'affraid': 'afraid', 'afterall': 'after all', 'agin': 'again',
    'alittle': 'a little', 'alrady': 'already', 'alread': 'already', 'alredy': 'already',
    'alrite': 'alright', 'altough': 'although', 'ammount': 'amount', 'anser': 'answer',
    'anwser': 'answer', 'anyhelp': 'any help', 'anyhoo': 'anyhow', 'anyhting': 'anything',
    'anyones': "anyone's", 'aobut': 'about', 'apci': 'acpi', 'apears': 'appears',
    'aplication': 'application', 'apparantly': 'apparently', 'appearence': 'appearance',
    'appriciate': 'appreciate', 'arounds': 'workarounds', 'arround': 'around',
    'availible': 'available', 'awnser': 'answer', 'azereus': 'azureus',
    'basicaly': 'basically', 'becasue': 'because', 'beeing': 'being', 'beggining': 'beginning',
    'beleve': 'believe', 'biggy': 'biggie', 'bizzare': 'bizarre', 'blahblah': 'blah blah',
    'broadcomm': 'broadcom', 'broswer': 'browser', 'celcius': 'celsius', 'cept': 'except',
    'channal': 'channel', 'chek': 'check', 'chk': 'check', 'choise': 'choice', 'clic': 'click',
    'cntrl': 'control', 'comands': 'commands', 'comman': 'command', 'commmand': 'command',
    'compatable': 'compatible', 'comptuer': 'computer', 'compy': 'computer',
    'conect': 'connect', 'controler': 'controller', 'controll': 'control', 'coool': 'cool',
    'coudl': 'could', 'coulda': 'could have', 'creat': 'create', 'customise': 'customize',
    'darnit': 'darn it', 'dchp': 'dhcp', 'debain': 'debian', 'definetely': 'definitely',
    'definetly': 'definitely', 'defult': 'default', 'dekstop': 'desktop',
    'dependecies': 'dependencies', 'destop': 'desktop', 'didnot': 'did not', 'dieing': 'dying',
    'dif': 'diff', 'differance': 'difference', 'differant': 'different',
    'differnce': 'difference', 'disapear': 'disappear', 'disrto': 'distro',
    'disrtos': 'distros', 'dissapear': 'disappear', 'dissapeared': 'disappeared',
    'dissapointed': 'disappointed', 'dkpg': 'dpkg', 'dled': 'downloaded',
    'dling': 'downloading', "doen't": "doesn't", 'doesnot': 'does not',
    'doesntwork': "doesn't work", 'doke': 'okay', 'dokey': 'okay', 'donot': 'do not',
    'donwload': 'download', 'dowload': 'download', 'downlaod': 'download', 'dpgk': 'dpkg',
    'easyer': 'easier', 'elp': 'help', 'encyption': 'encryption',
    'enlightment': 'enlightenment', 'enought': 'enough', 'enuf': 'enough', 'eror': 'error',
    'errror': 'error', 'evenin': 'evening', 'everythin': 'everything',
    'everythings': "everything's", 'everyting': 'everything', 'eveything': 'everything',
    'evry': 'every', 'evryone': 'everyone', 'evrything': 'everything', 'exaclty': 'exactly',
    'excactly': 'exactly', 'excatly': 'exactly', 'excelent': 'excellent', 'exept': 'except',
    'existance': 'existence', 'experiance': 'experience', 'explaination': 'explanation',
    'extentions': 'extensions', 'extremly': 'extremely', 'ffox': 'firefox',
    'firfox': 'firefox', 'flgrx': 'fglrx', 'fraid': 'afraid', 'franais': 'french',
    'freezed': 'froze', 'freind': 'friend', 'frickin': 'freaking', 'frm': 'from',
    'genious': 'genius', 'gimmie': 'gimme', 'gn': 'good night', 'gnight': 'goodnight',
    'goign': 'going', 'goodmorning': 'good morning', 'gota': 'gotta', 'gots': 'got',
    'gratz': 'congrats', 'grt': 'great', 'grup': 'group', 'gud': 'good', 'habbit': 'habit',
    'hadnt': "hadn't", 'haev': 'have', 'hardisk': 'hard disk', 'harware': 'hardware',
    'haveing': 'having', 'haveto': 'have to', 'havin': 'having', 'heared': 'heard',
    'heloo': 'hello', 'helpin': 'helping', 'heyy': 'hey', 'hilight': 'highlight',
    'hlep': 'help', 'hlp': 'help', 'horay': 'hooray', 'howcome': 'how come', 'howd': "how'd",
    'howso': 'how so', 'htere': 'there', 'htink': 'think', 'hwat': 'what',
    'hybernate': 'hibernate', 'hye': 'hey', 'idd': 'indeed', 'ideea': 'idea',
    'idont': "I don't", 'ihave': 'I have', 'iknow': 'I know', 'immediatly': 'immediately',
    'imporntant': 'important', 'inbetween': 'in between', 'indeedy': 'indeed',
    'independant': 'independent', 'inglish': 'english', 'insall': 'install',
    'instaling': 'installing', 'installl': 'install', 'instlal': 'install',
    'intalled': 'installed', 'inte': 'intel', 'intell': 'intel', 'intergrated': 'integrated',
    'interpid': 'intrepid', 'intersting': 'interesting', 'intresting': 'interesting',
    'irrsi': 'irssi', 'isee': 'I see', 'isntalling': 'installing', 'istall': 'install',
    'itd': "it'd", 'ithink': 'I think', 'iwth': 'with', 'jup': 'yup', 'kiddin': 'kidding',
    'kindof': 'kind of', 'knowledgable': 'knowledgeable', 'kow': 'know', 'labtop': 'laptop',
    'lappie': 'laptop', 'letme': 'let me', 'lik': 'like', 'linky': 'link', 'litle': 'little',
    'loged': 'logged', 'loging': 'logging', 'lok': 'look', 'lolwut': 'lol what',
    'lscpi': 'lspci', 'lunix': 'linux', 'mabey': 'maybe', 'mabye': 'maybe', 'makin': 'making',
    'managment': 'management', 'mayb': 'maybe', 'mebbe': 'maybe', 'mee': 'me',
    'messanger': 'messenger', 'messin': 'messing', 'messsage': 'message',
    'missread': 'misread', 'missunderstood': 'misunderstood', 'mke': 'make', 'mmkay': 'mkay',
    'moniter': 'monitor', 'mor': 'more', 'mroe': 'more', 'mutch': 'much',
    'natilus': 'nautilus', 'nautilius': 'nautilus', 'nautlius': 'nautilus', 'neato': 'neat',
    'neccesary': 'necessary', 'neccessary': 'necessary', 'necesary': 'necessary',
    'neeed': 'need', 'netowrk': 'network', 'newbe': 'newbie', 'newbee': 'newbie',
    'newby': 'newbie', 'nividia': 'nvidia', 'no1': 'no one', 'nonono': 'no no',
    'noobie': 'newbie', 'nother': 'another', 'nouse': 'no use', 'nowdays': 'nowadays',
    'nuff': 'enough', 'occuring': 'occurring', 'offcourse': 'of course', 'offical': 'official',
    'okee': 'okay', 'okej': 'okay', 'olds': 'old', 'onw': 'own', 'othe': 'other',
    'ouput': 'output', 'pacakge': 'package', 'pakage': 'package', 'partation': 'partition',
    'partioned': 'partitioned', 'partioning': 'partitioning', 'pasword': 'password',
    'pavillion': 'pavilion', 'peice': 'piece', 'peopel': 'people', 'perfer': 'prefer',
    'permision': 'permission', 'permisions': 'permissions', 'pidgen': 'pidgin',
    'plase': 'please', 'pleae': 'please', 'pluggin': 'plugin', 'pluging': 'plugging',
    'poeple': 'people', 'prblem': 'problem', 'preformance': 'performance',
    'premissions': 'permissions', 'prety': 'pretty', 'probally': 'probably',
    'probaly': 'probably', 'probem': 'problem', 'probleme': 'problem', 'problm': 'problem',
    'problme': 'problem', 'proccess': 'process', 'proggy': 'program', 'programms': 'programs',
    'promt': 'prompt', 'propietary': 'proprietary', 'proprietry': 'proprietary',
    'propritary': 'proprietary', 'questoin': 'question', 'qustion': 'question',
    'readin': 'reading', 'reall': 'really', 'realplay': 'realplayer',
    'reccommend': 'recommend', 'recieving': 'receiving', 'recomendations': 'recommendations',
    'rediculous': 'ridiculous', 'refered': 'referred', 'reffering': 'referring',
    'registerd': 'registered', 'relase': 'release', 'relly': 'really',
    'removeable': 'removable', 'repeate': 'repeat', 'replys': 'replies',
    'responce': 'response', 'respository': 'repository', 'restriced': 'restricted',
    'righty': 'right', 'rockin': 'rocking', 'runnig': 'running', 'runnign': 'running',
    'rythembox': 'rhythmbox', 'samething': 'the same thing', 'sayd': 'said', 'scren': 'screen',
    'screwd': 'screwed', 'seach': 'search', 'seee': 'see', 'seperated': 'separated',
    'seperately': 'separately', 'sho': 'show', 'shold': 'should', 'shoul': 'should',
    'shoulda': 'should have', 'shuld': 'should', 'sicne': 'since', 'sistem': 'system',
    'skillz': 'skills', 'sollution': 'solution', 'somwhere': 'somewhere', 'sooooo': 'so',
    'sorr': 'sorry', 'sorrry': 'sorry', 'sortof': 'sort of', 'statment': 'statement',
    'sucess': 'success', 'sugest': 'suggest', 'sugestion': 'suggestion',
    'sugestions': 'suggestions', 'sumthing': 'something', 'suport': 'support',
    'suported': 'supported', 'supose': 'suppose', 'suposed': 'supposed', 'suprise': 'surprise',
    'sweeet': 'sweet', 'swith': 'switch', 'synatpic': 'synaptic', 'synpatic': 'synaptic',
    'syste': 'system', 'tahnks': 'thanks', 'tak': 'yes', 'tanx': 'thanks', 'tellme': 'tell me',
    'teminal': 'terminal', 'termianl': 'terminal', 'thak': 'thank', 'thakns': 'thanks',
    'thang': 'thing', 'thankee': 'thanks', 'thankies': 'thanks', 'thanky': 'thanks',
    'thankz': 'thanks', 'thanls': 'thanks', 'thanxs': 'thanks', 'thas': "that's",
    'thast': "that's", 'thatd': "that'd", 'thatll': "that'll", 'thatnks': 'thanks',
    'thax': 'thanks', 'thet': 'that', 'thi': 'this', 'thik': 'think', 'thnkx': 'thanks',
    'thoes': 'those', 'thos': 'those', 'thot': 'thought', 'throught': 'throughout',
    'thta': 'that', 'thts': "that's", 'thxs': 'thanks', 'tihnk': 'think', 'todays': "today's",
    'tommorow': 'tomorrow', 'tomorow': 'tomorrow', 'transfering': 'transferring',
    'truely': 'truly', 'tryied': 'tried', 'tryign': 'trying', 'tryng': 'trying',
    'trys': 'tries', 'txs': 'thanks', 'tyou': 'to you', 'typ': 'type', 'tyring': 'trying',
    'ubantu': 'ubuntu', 'ubntu': 'ubuntu', 'ubnutu': 'ubuntu', 'ububtu': 'ubuntu',
    'ubugtu': 'ubuntu', 'ubunu': 'ubuntu', 'ubuto': 'ubuntu', 'uhoh': 'uh oh',
    'unbutu': 'ubuntu', 'undestand': 'understand', 'ununtu': 'ubuntu', 'ure': "you're",
    'urself': 'yourself', 'usally': 'usually', 'useage': 'usage', 'usin': 'using',
    'ususally': 'usually', 'utube': 'youtube', 'verison': 'version', 'vey': 'very',
    'wahts': "what's", 'wasent': "wasn't", 'wath': 'watch', 'welcom': 'welcome',
    'welll': 'well', 'werd': 'word', 'werid': 'weird', 'whas': 'was', 'whast': 'what',
    'whe': 'when', 'whith': 'with', 'whoah': 'whoa', 'whould': 'would', 'whre': 'where',
    'whta': 'what', 'whts': "what's", 'wiat': 'wait', 'willl': 'will', 'windoes': 'windows',
    'windos': 'windows', 'windowz': 'windows', 'windoz': 'windows', 'windwos': 'windows',
    'wireles': 'wireless', 'wirelss': 'wireless', 'witht': 'with', 'wll': 'will',
    'wokr': 'work', 'wonderfull': 'wonderful', 'wor': 'work', 'workd': 'worked',
    'workign': 'working', 'worky': 'work', 'worng': 'wrong', 'woth': 'with', 'wots': "what's",
    'woul': 'would', 'woulda': 'would have', 'writting': 'writing', 'wrk': 'work',
    'wtg': 'way to go', 'wud': 'would', 'wuts': "what's", 'xbuntu': 'xubuntu', 'xcfe': 'xfce',
    'yaeh': 'yeah', 'yesss': 'yes', 'yh': 'yeah', 'yor': 'your', 'yourfile': 'your file',
    'yout': 'your', 'yoy': 'you', 'ypu': 'you', 'yse': 'yes', 'yuppers': 'yep', 'yur': 'your',

    # --- batch 6: finishing the 20-49 occurrence band ---
    'achive': 'achieve', 'alway': 'always', 'amorok': 'amarok', 'anway': 'anyway',
    'anyoen': 'anyone', 'anyon': 'anyone', 'anyother': 'any other',
    'apperance': 'appearance', 'areyou': 'are you', 'aviable': 'available',
    'avoide': 'avoid', 'beacuse': 'because', 'bein': 'being', 'bery': 'very',
    'blabla': 'blah blah', 'cannt': 'cannot', 'changeing': 'changing',
    'channle': 'channel', 'chech': 'check', 'conenction': 'connection',
    'curser': 'cursor', 'datas': 'data', 'dayo': 'day', 'dektop': 'desktop',
    'delet': 'delete', 'developement': 'development', 'didint': "didn't",
    'dissapears': 'disappears', 'doent': "doesn't", 'doest': "doesn't", 'doki': 'okay',
    'doubtfull': 'doubtful', 'dows': 'does', 'doyou': 'do you', 'easially': 'easily',
    'englis': 'english', 'everone': 'everyone', 'everyones': "everyone's",
    'eveyone': 'everyone', 'experince': 'experience', 'eys': 'eyes',
    'familliar': 'familiar', 'finaly': 'finally', 'finde': 'find', 'foward': 'forward',
    'gday': 'good day', 'geeze': 'geez', 'gj': 'good job', 'goole': 'google',
    'grats': 'congrats', 'handbreak': 'handbrake', 'happended': 'happened',
    'helpp': 'help', 'hilighted': 'highlighted', 'ight': 'alright',
    'installin': 'installing', 'installtion': 'installation',
    'intalling': 'installing', 'intsall': 'install', 'irsii': 'irssi', 'itz': "it's",
    'iunno': 'I dunno', 'jave': 'java', 'jeah': 'yeah', 'jejeje': 'haha',
    'konquerer': 'konqueror', 'likley': 'likely', 'linix': 'linux',
    'logitec': 'logitech', 'loking': 'looking', 'lovin': 'loving', 'lul': 'lol',
    'lvl': 'level', 'mabe': 'maybe', 'mdr': 'lol', 'minuts': 'minutes',
    'nautilis': 'nautilus', 'ne1': 'anyone', 'neccessarily': 'necessarily',
    'nfts': 'ntfs', 'nooby': 'newbie', 'noup': 'nope', 'ntsf': 'ntfs',
    'nubbie': 'newbie', 'nup': 'nope', 'nuthin': 'nothing', 'nvdia': 'nvidia',
    'okidoki': 'okey dokey', 'okz': 'okay', 'optimised': 'optimized',
    'passowrd': 'password', 'pitty': 'pity', 'playin': 'playing', 'plese': 'please',
    'portugese': 'portuguese', 'pplz': 'people', 'ppoe': 'pppoe', 'prolem': 'problem',
    'propriatary': 'proprietary', 'queston': 'question', 'quetion': 'question',
    'reallly': 'really', 'reciever': 'receiver', 'reinstal': 'reinstall',
    'relevent': 'relevant', 'repositries': 'repositories', 'reseting': 'resetting',
    'reso': 'resolution', 'resoultion': 'resolution', 'rhythymbox': 'rhythmbox',
    'rror': 'error', 'runned': 'ran', 'runnning': 'running', 'rythymbox': 'rhythmbox',
    'sattelite': 'satellite', 'serach': 'search', 'shld': 'should', 'shuold': 'should',
    'siad': 'said', 'sitll': 'still', 'skool': 'school', 'smae': 'same',
    'sofware': 'software', 'somehwere': 'somewhere', 'somtimes': 'sometimes',
    'soooooo': 'so', 'speack': 'speak', 'succesfully': 'successfully', 'sudu': 'sudo',
    'temrinal': 'terminal', 'tenx': 'thanks', 'termial': 'terminal',
    'termina': 'terminal', 'thare': 'there', 'theese': 'these', 'theyve': "they've",
    'thign': 'thing', 'thigns': 'things', 'thnak': 'thanks', 'thougt': 'thought',
    'thsi': 'this', 'trie': 'try', 'ture': 'true', 'tweek': 'tweak',
    'ubunty': 'ubuntu', 'undertand': 'understand', 'wazzup': "what's up",
    'weired': 'weird', 'whaat': 'what', 'whcih': 'which', 'whereever': 'wherever',
    'whut': 'what', 'woh': 'wow', 'wonderin': 'wondering', 'wron': 'wrong',
    'wuestion': 'question', 'yeppers': 'yep', 'yessss': 'yes', 'yey': 'yay',
    'yoiu': 'you', 'yoo': 'you', 'yopu': 'you', 'youy': 'you', 'yuou': 'you',

    # --- batch 7: closing the >=50 occurrence gap ---
    'acc': 'account', "arn't": "aren't", "avi's": 'avis', 'bb': 'bye bye',
    'bw': 'bandwidth', "does'nt": "doesn't", 'dokie': 'okay', 'dup': 'duplicate',
    'elses': "else's", 'erro': 'error', 'exp': 'experience', 'freakin': 'freaking',
    'fuckin': 'fucking', 'fw': 'forward', 'hax': 'hacks', "hd's": 'hard drives',
    "i'am": 'I am', "i'l": "I'll", 'ik': 'I know', "isp's": 'isps', 'jep': 'yep',
    'kbs': 'kbps', 'lawl': 'lol', 'loggin': 'logging', 'lolol': 'lol', 'mh': 'hmm',
    'mm': 'hmm', 'mmh': 'hmm', 'mmk': 'okay', 'mmm': 'hmm', 'newbs': 'newbies',
    'nn': 'night night', 'nono': 'no no', 'offence': 'offense', 'owh': 'oh',
    'oyu': 'you', 'sais': 'says', 'synaptec': 'synaptic', "tty's": 'ttys',
    'tx': 'thanks', 'upto': 'up to', 'urgh': 'ugh', "uuid's": 'uuids', "vm's": 'vms',
    'wellcome': 'welcome', 'wil': 'will', "ya'll": "y'all", 'yesh': 'yes', 'yu': 'you',

    # --- batch 8: reclassified out of JARGON_TERMS as plain typos ---
    'chater': 'chatter', 'chating': 'chatting', 'quited': 'quit',
    'quiter': 'quitter', 'quiting': 'quitting',
}


def canonicalize_word(wl):
    """Variable-length filler/laugh/interjection families -- pure
    character-composition rules, since a fixed dict can't cover
    open-ended-length variants. `wl` must already be lowercased. Returns
    the canonical spelling, or None if `wl` doesn't match any family."""
    if set(wl) <= {'h', 'a'} and 'ha' in wl: return 'haha'   # laughing
    if set(wl) <= {'h', 'e'} and 'heh' in wl: return 'hehe'  # laughing
    if re.fullmatch(r'hm+', wl): return 'hmm'                # thinking
    if re.fullmatch(r'oh+', wl): return 'oh'
    if re.fullmatch(r'o{2,}h*', wl): return 'ooh'             # sudden realization
    if re.fullmatch(r'uh*m+', wl): return 'um'
    if re.fullmatch(r'uh+', wl): return 'uh'
    if re.fullmatch(r'gr+', wl): return 'grr'
    if re.fullmatch(r'ah+', wl): return 'ah'
    if re.fullmatch(r'hi+', wl): return 'hi'
    if re.fullmatch(r'eh+', wl): return 'eh'
    if re.fullmatch(r'ew+', wl): return 'ew'
    if re.fullmatch(r'aw+e*', wl): return 'aw'
    if re.fullmatch(r'wa+h+', wl): return 'wah'                # crying
    if re.fullmatch(r'hello+', wl): return 'hello'
    if re.fullmatch(r'argh(?:gh)*h*', wl): return 'argh'       # frustration
    if re.fullmatch(r'a{2,}h+', wl): return 'aah'              # realization/surprise/sympathy/pleasure
    if re.fullmatch(r'to{2,}', wl): return 'too'
    if re.fullmatch(r'na+h+', wl): return 'nah'
    if re.fullmatch(r'lo+l+', wl): return 'lol'
    if re.fullmatch(r'no+', wl): return 'no'
    if re.fullmatch(r'erm+', wl): return 'erm'                 # awkwardness / pause to think
    return None


_FAMILY_ALT = '|'.join([
    r'[ha]*ha[ha]*', r'[he]*heh[he]*', r'h[m]+', r'o[h]+', r'o{2,}[h]*',
    r'u[h]*m+', r'u[h]+', r'g[r]+', r'a[h]+', r'h[i]+', r'e[h]+', r'e[w]+', r'a[w]+e*',
    r'wa+h+', r'hello+', r'argh(?:gh)*h*', r'a{2,}h+', r'to{2,}',
    r'na+h+', r'lo+l+', r'no+', r'erm+',
])

# Candidate-catching regex: only words that could plausibly need correction
# get matched at all (fast, vectorized, C-level regex filtering) -- the
# Python callback below does the exact classification only for those hits.
#
# The trailing (?!') guards against a specific collision: CORRECTIONS has
# "doesn": "doesn't" and "didn": "didn't" (to catch cases upstream where the
# apostrophe+t got stripped off a contraction). But apostrophe is not a \w
# character, so \b fires right before it -- meaning \bdoesn\b also matches
# the "doesn" inside an ALREADY-correct "doesn't", turning it into
# "doesn't't". The negative lookahead stops the match before it ever
# consumes a token that's immediately followed by an apostrophe, so an
# intact "doesn't"/"didn't" is left alone.
_CANDIDATE_PATTERN = re.compile(
    r"\b(?:" + '|'.join(
        [re.escape(k) for k in sorted(CORRECTIONS, key=len, reverse=True)] + [_FAMILY_ALT]
    ) + r")\b(?!')",
    re.IGNORECASE,
)


def _replace_word(m):
    w = m.group(0)
    wl = w.lower()
    if wl in CORRECTIONS:
        return CORRECTIONS[wl]
    canon = canonicalize_word(wl)
    return canon if canon else w


def normalize_chat_shorthand(series):
    """Apply CORRECTIONS + the filler/laugh/interjection families to a
    text_cleaned Series. Vectorized (one compiled regex, one pass) -- no
    need to parallelize this the way the per-row lexicon-matching phases
    are, same reasoning apply_lexicons.py's anonymize_structural uses."""
    return series.str.replace(_CANDIDATE_PATTERN, _replace_word, regex=True)


# --- Latin-extended (accented Romance-language) corrections + deja vu phrase
# normalization, from the extended-Latin-bucket review. Distinct from
# CORRECTIONS/canonicalize_word above -- those are ASCII-only English chat
# shorthand and never see accented input (canonicalize_word('héhé') -> None,
# confirmed: {'h','é'} isn't a subset of {'h','e'}) -- but same purpose: fix
# known accented/decorated spellings before anything falls through to a
# generic catch-all.
LATIN_WORD_CORRECTIONS = {
    "thankś": "thanks", "networksß": "networks", "iḿ": "I'm", "iĺl": "I'll",
    "itś": "it's", "helló": "hello", "vírus": "virus", "vídeos": "videos",
    "vidéo": "video", "runß": "run", "thatß": "that", "thinkç": "think",
    "ók": "ok", "i'äm": "I am", "fæces": "feces", "âss": "ass", "wíth": "with",
    "ｒｅｓｏｌｕｔｉｏｎ": "resolution", "installé": "install", "sörry": "sorry",
    "héhé": "hehe",
    # upside-down/flipped-text joke spellings (turned-letter Unicode block)
    "sǝdʎʇ": "types", "sʎɐʍlɐ": "always", "uʍop": "down", "ubuntuß": "ubuntu",
    "whatś": "what's", "ǝpısdn": "upside", "ǝɯɐu": "name", "ʎuɐ": "any", "ʎɯ": "my",
}

_LATIN_CORRECTIONS_PATTERN = re.compile(
    r'\b(?:' + '|'.join(re.escape(k) for k in sorted(LATIN_WORD_CORRECTIONS, key=len, reverse=True)) + r')\b',
    re.IGNORECASE,
)

# deja/déjà/dejà/déja/dejá vu, any accent combo, with a space, a hyphen, or
# nothing between the two words (dejavu, déjàvu, deja-vu, deja vu)
DEJA_VU_RE = re.compile(r'\bd[eé]j[aàá][\s-]*vu\b', re.IGNORECASE)


def apply_latin_corrections(series):
    """Fixed half of the extended-Latin cleanup: deja vu normalization +
    the explicit accent/decorated-token corrections. Does NOT include a
    generic catch-all -- that depends on a vocabulary set computed live
    from a specific no_match_df run, so it has to stay in the notebook
    rather than becoming a static module constant."""
    out = series.str.replace(DEJA_VU_RE, 'deja vu', regex=True)
    out = out.str.replace(_LATIN_CORRECTIONS_PATTERN, lambda m: LATIN_WORD_CORRECTIONS[m.group(0).lower()], regex=True)
    return out