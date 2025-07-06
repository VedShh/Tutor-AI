from flask import Flask

app = Flask(__name__)
@app.route("/hello")
def helloworld():
    return "Hello World - Vedant1!"
if __name__ == "__main__":
    app.run()