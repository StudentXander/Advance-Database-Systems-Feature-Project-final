import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from app import Candidate, JobPosition, Matching
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

def train_model():
    # Load data from the database
    engine = create_engine('sqlite:///job_match.db')
    Base.metadata.bind = engine
    DBSession = sessionmaker(bind=engine)
    session = DBSession()

    candidates = session.query(Candidate).all()
    job_positions = session.query(JobPosition).all()
    matches = session.query(Matching).all()

    # Preprocess the data
    candidate_df = pd.DataFrame([(c.IDcandidate, c.last_name, c.first_name, c.middle_name,
                                  c.name_extension, c.Skills, c.Experience, c.Education,
                                  c.contactno, c.email) for c in candidates],
                                columns=['IDcandidate', 'last_name', 'first_name', 'middle_name',
                                         'name_extension', 'Skills', 'Experience', 'Education',
                                         'contactno', 'email'])

    job_position_df = pd.DataFrame([(j.JobID, j.Position, j.Description, j.Required_Skills)
                                    for j in job_positions],
                                   columns=['JobID', 'Position', 'Description', 'Required_Skills'])

    match_df = pd.DataFrame([(m.MatchID, m.JobID, m.CandidateID, m.MatchScore, m.Status)
                             for m in matches],
                            columns=['MatchID', 'JobID', 'CandidateID', 'MatchScore', 'Status'])

    # Merge candidate and job_position data
    merged_df = pd.merge(match_df, candidate_df, on='IDcandidate', how='inner')
    merged_df = pd.merge(merged_df, job_position_df, on='JobID', how='inner')

    # Feature engineering (you may need to customize this based on your specific data)
    merged_df['Skills_match'] = merged_df.apply(lambda row: row['Skills'] in row['Required_Skills'], axis=1)

    # Encode categorical variables
    label_encoder = LabelEncoder()
    merged_df['Education'] = label_encoder.fit_transform(merged_df['Education'])

    # Select features and target variable
    X = merged_df[['Experience', 'Education', 'Skills_match']]
    y = merged_df['Status']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Model Accuracy: {accuracy}')

# Uncomment the line below to insert sample data into the database
# insert_sample_data()

# Uncomment the line below to train the model
# train_model()
