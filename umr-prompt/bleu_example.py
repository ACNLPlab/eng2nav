import evaluate

# Example sentences
# reference = "Yiską́ągo abnídą́ą índa dooleeł nááhodoo’niid."
# hypothesis = "Dóó bił naashá, níłch’i azhíígo háwíí dą́ą yî’í nánígo."
reference = "’Áłtsé Hastiin dóó ’Áłtsé ’Asdzą́ą́ woliiínda Yoołgaii ’Asdzą́ą́ó’Asdzą́ą́ dlee ’ahíjiikaigo nidzíiztą́ jin"
hypothesis = "’Áłtsé Hastiin dóó ’Áłtsé Asdzáán, dóó Yoołgai Asdzáán dóó Asdzáán Nádleehé wolyéígíí daníyáá’go daayisdá"

# Load BLEU metric
bleu = evaluate.load("bleu")

# Compute BLEU
result = bleu.compute(predictions=[hypothesis], references=[[reference]])

# BLEU is between 0 and 1, multiply by 100 for readability
print("BLEU score:", result["bleu"] * 100)

