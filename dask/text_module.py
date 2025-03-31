#!/usr/bin/env python3

'''
These module provide functionality for adding text to a Word document.
'''

__author__ = "V.Pazenko"
__version__ = 1.0

def add_text_to_doc(doc, section):
    '''
    Input: 1. doc = Word document object
           2. section = string with section name

    Function add a text to a Word document according to the given section name.

    Output: None, add text to the document (doc)
    '''

    text_memo_exec = 'As we can see in the tables above, the execution of transformations ' \
        'in Pandas takes less time than in Dask, but Dask can store information in memory partially ' \
        'and process it in parallel and only as much as needed, which allows it to handle large datasets.' \
        'Probably the amount of data is too small to clearly demonstrate the speed of parallel operation.'

    text_model = 'The models results show that despite the longer train/predict Dask model time, '\
        'many indicators are higher for this model.\n'\
        'However, here we should make some remarks: the function dask_ml.model_selection.train_test_split '\
        'does not have an argument stratification, which could affect the final result because our dataframe '\
        'is unbalanced. The main work was done on a Windows computer, but '\
        '"Windows is not officially supported for dask/xgboost".'


    if section == 'Execution time':
        doc.add_paragraph("\n")
        # doc.add_paragraph(f"{section}:")  # Добавляем заголовок раздела
        doc.add_paragraph(text_memo_exec)  # Добавляем сам текст
        doc.add_paragraph("\n")  # Добавляем отступ для читаемости


    elif section == 'Model':
        doc.add_paragraph("\n")
        # doc.add_paragraph(f"{section}:")  # Добавляем заголовок раздела
        doc.add_paragraph(text_model)  # Добавляем сам текст
        doc.add_paragraph("\n")  # Добавляем отступ для читаемости

