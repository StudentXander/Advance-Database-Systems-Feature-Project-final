from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from database_utils import train_model, predict_match  # Import from the new module
import os

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///job_match.db'
db = SQLAlchemy(app)

Base = declarative_base()

class Candidate(Base):
    __tablename__ = 'candidates'
    IDcandidate = db.Column(db.Integer, primary_key=True, autoincrement=True)
    last_name = db.Column(db.String(255))
    first_name = db.Column(db.String(255))
    middle_name = db.Column(db.String(255))
    name_extension = db.Column(db.String(255))
    Skills = db.Column(db.String(255))
    Experience = db.Column(db.Float)
    Education = db.Column(db.String(255))
    contactno = db.Column(db.String(45))
    email = db.Column(db.String(45))

class JobPosition(Base):
    __tablename__ = 'job_positions'
    JobID = db.Column(db.Integer, primary_key=True, autoincrement=True)
    Position = db.Column(db.String(255))
    Description = db.Column(db.String(255))
    Required_Skills = db.Column(db.String(255))  # You can adjust the data type if needed

class Matching(Base):
    __tablename__ = 'matching'
    MatchID = db.Column(db.Integer, primary_key=True, autoincrement=True)
    JobID = db.Column(db.Integer)
    CandidateID = db.Column(db.Integer)
    MatchScore = db.Column(db.Float)
    Status = db.Column(db.String(50))

db.create_all()

@app.route('/')
def index():
    candidates = Candidate.query.all()
    job_positions = JobPosition.query.all()
    matches = Matching.query.all()
    return render_template('index.html', candidates=candidates, job_positions=job_positions, matches=matches)

@app.route('/train_model')
def train_model_route():
    train_model()  # Call the function to train the machine learning model
    return "Model trained successfully!"

@app.route('/predict_match')
def predict_match_route():
    matches = predict_match()  # Call the function to make predictions
    return f"Predicted Matches: {matches}"

if __name__ == '__main__':
    app.run(debug=True)
