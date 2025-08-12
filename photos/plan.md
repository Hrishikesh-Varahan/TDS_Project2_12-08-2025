# TDS_project2
AI based Data Analyst


### Approch1: 

1. give question.txt, and any other file(csv, img etc) to llm and ask for code and libraries to execute.

2. execute that code in python repl.

Problem: the llm writes code assuming the names on itself which gave raise to too many errors.

### Approch-2:

1. Step first is same. but instead of just whole code we are asking to write code to scrap or read all the data that is avaialable(or given).

2. now gave that question statement and the ouput data to llm again and ask for possible solution which includes:
    - if the problem is small, or chatgpt can answer that  `go ahead and do that` if not,
    - now provide the code for that and we will execute it and answer the question accordingly.


### Approch-3(little modification in approch-2):
 - instead of handing over all the scraped data and question statement to llm again. We done this:
    -   ask first llm to give code for scraping the data and created metadata.txt file that has:
        - metadata of scraped data.
        - important file paths and description of file.
        - asked questions .
        - json format for giving answers.

 - Second llm call gets all the data file paths along with metadata. so this llm tries to generated code for solving them and returning a json based on provided format.

`Note: Sometimes the llm generated code has little mistakes. so, i decided to give llm 3 chances to correct that.`
 
