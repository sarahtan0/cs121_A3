from flask import Flask, request
from index_search import search
# from flask_wtf.csrf import CSRFProtect

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    html_content = "<h1>Results</h1>"
    results = []
    query = ""

    if request.method == 'POST':
        query = request.form.get('query', '')
        if not query:
            return "NOT VALID QUERY"

        results = search(query)  # Run the search function

    for result in results:
        html_content+=f"<a href='{result}'>{result}</a><br>"

    if results:
        return html_content
    else:
        return f'''
        <h1>LeSearch</h1>
        <form action="/" method="POST">
            <input type="text" name="query" placeholder="Enter search term" value="{query}" required>
            <button type="submit">Search</button>
        </form>
        ''' # Initial page load

if __name__ == '__main__':
    app.run(debug=True)