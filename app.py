from flask import Flask, request, render_template
from index_search import search
# from flask_wtf.csrf import CSRFProtect

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    results = []
    query = ""

    if request.method == 'POST':
        query = request.form.get('query', '')
        if not query:
            return "NOT VALID QUERY"

        results = search(query)  # Run the search function

    if results:
        return render_template('results.html', results=results)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)