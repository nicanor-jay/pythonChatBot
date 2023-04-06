# NLP Chatbot
A simple NLP chatbot made with Python (NLTK, scipy, scikit) during my time at university.

## UI Loop
- The user is first prompted for a name
- The user is prompted to enter a query.
- The chatbot responds to the query with the most appropriate reply it finds.
- The user is then propted for the next query, if 'STOP' is entered, the loop ends.
- 
## How it works
The chatbot works off of giving the program CSV files based on the intent behind each query that you would want to process. My project includes the below 'intent' routes:
- Question-answer
- Small-talk 
- Name-get
- Name-set

A classifier is then trained to these in order to denote what type of 'intention' is behind any user query.

### Question-answer & Small-talk
If the query is classified as a "question-answer" prompt, it will then calculate a similarity index between the query, and the list of queston-answer pairs within the 'data.csv' files. If the similarity index is above a certain threshold, the respective answer is returned.
This is also true for any queries classified as 'small-talk', along with there being multiple of the same query, but with different responses, which means that repetitive questions can be answered with different, yet appropriate, replies.

### Name-get & Name-set
If the query is classified as a 'name-get', the chatbot simply returns the users name back to them

However, if the "Name-set" intent is detected, there are two ways that it can be done:
1. Explicitly
    - This is done by entering the name once the ChatBot has asked you for your new name
    ![image](https://user-images.githubusercontent.com/95185431/230493174-ca8f644f-304c-4785-869a-ffd79806b124.png)

2. Implicitly
    - This is done by using POS(Part-Of-Speech) tagging in order to try and discern a name from a string, such as:
    ![image](https://user-images.githubusercontent.com/95185431/230493424-c8895231-7edd-4bc4-a19d-042eeb4added.png)
