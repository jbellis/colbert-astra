from flask import Flask, request, render_template_string
from serve import retrieve_ada, retrieve_colbert

app = Flask(__name__)

# Updated HTML template for displaying results as sections with titles and bodies
HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <!-- Bootstrap CSS (using a public CDN) -->
  <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css">
  <title>Retrieve Models Interface</title>
</head>
<body>
<div class="container">
  <h2>DPR vs ColBERT</h2>
  <form method="post">
    <div class="form-group">
      <label for="queryInput">Enter your query:</label>
      <input type="text" class="form-control" id="queryInput" name="query" required>
    </div>
    <button type="submit" class="btn btn-primary">Submit</button>
  </form>
  {% if results %}
  <div class="row">
    <div class="col-md-6">
      <h3>DPR (ada002) Results</h3>
      {% for result in results.ada %}
        <h5>{{ result.title }}</h5>
        <p>{{ result.body }}</p>
      {% endfor %}
    </div>
    <div class="col-md-6">
      <h3>ColBERT Results</h3>
      {% for result in results.colbert %}
        <h5>{{ result.title }}</h5>
        <p>{{ result.body }}</p>
      {% endfor %}
    </div>
  </div>
  {% endif %}
</div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def index():
    results = None
    if request.method == 'POST':
        query = request.form['query']
        ada_results = retrieve_ada(query)  # Ensure this returns a list of dicts with 'title' and 'body'
        colbert_results = retrieve_colbert(query)  # Ensure this returns a list of dicts with 'title' and 'body'
        results = {'ada': ada_results, 'colbert': colbert_results}
    return render_template_string(HTML_TEMPLATE, results=results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
