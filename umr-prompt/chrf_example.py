import evaluate 

#reference = "Yiską́ągo abnídą́ą índa dooleeł nááhodoo’niid." 
#hypothesis = "Dóó bił naashá, níłch’i azhíígo háwíí dą́ą yî’í nánígo." 
reference = "’Áłtsé Hastiin dóó ’Áłtsé ’Asdzą́ą́ wolii ’índa Yoołgaii ’Asdzą́ą́ óó’Asdzą́ą́ dleehé ’ahíjiikaigo nidzíiztą́ jiní" 

hypothesis = "Áłtsé Hastiin dóó Áłtsé Asdzáán, dóó Yoołgai Asdzáán dóó Asdzáá Nádleehé danáályaa’go da sidá." 

chrf = evaluate.load("chrf") 

result = chrf.compute(predictions=[hypothesis], references=[reference]) 

print("CHRF score:", result["score"])
