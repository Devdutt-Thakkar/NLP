from flask import Flask, request, render_template
from transformers import pipeline

app = Flask(__name__)

#make pipelines
ner = pipeline("ner", grouped_entities = True)
summerizer = pipeline("summarization")
translator = pipeline("translation", model = "Helsinki-NLP/opus-mt-fr-en")
generational = pipeline("text-generation")
sentimental = pipeline("sentiment-analysis")


@app.route("/",methods = ['POST','GET'])
def index():
    sent = None
    gen = None
    trans = None
    neri = None
    summ = None
    if request.method == 'POST':
        user_input = request.form.get('user_input')
        task = request.form.get('task')

        if task == 'sentiment':
            sent = sentimental(user_input)
            return render_template('pipeline.html', sent= sent)
        elif task == 'generation':
            gen = generational(user_input)
            return render_template('pipeline.html', gen= gen)
        elif task == 'translation':
            trans = translator(user_input)
            return render_template('pipeline.html', trans= trans)
        elif task == 'summarization':
            summ = summerizer(user_input)
            return render_template('pipeline.html', summ= summ)
        elif task == 'named_entity_recognition':
            neri = ner(user_input)
            return render_template('pipeline.html', neri= neri)

    return render_template('pipeline.html')
if __name__=="__main__":
    app.run(debug = True)