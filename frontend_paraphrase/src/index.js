import React, { useState, useEffect } from 'react';
import axios from 'axios';

function ParaphraseForm() {
  const [inputText, setInputText] = useState('');
  const [outputTexts, setOutputTexts] = useState([]);

  const handleChange = (event) => {
    setInputText(event.target.value);
  };

  const handleSubmit = (event) => {
    event.preventDefault();

    // send a request to the Django backend with the input text
    axios.get('http://localhost:8000/paraphrase', {
      params: {
        input_text: inputText,
      }
    })
      .then(response => {
        setOutputTexts(response.data.paraphrases);
      })
      .catch(error => {
        console.log(error);
      });
  };

  return (
    <div>
      <form onSubmit={handleSubmit}>
        <label>
          Input Text:
          <input type="text" value={inputText} onChange={handleChange} />
        </label>
        <input type="submit" value="Paraphrase" />
      </form>
      <ul>
        {outputTexts.map((text, index) => (
          <li key={index}>
            {text[0]} (Similarity Score: {text[1]})
          </li>
        ))}
      </ul>
    </div>
  );
}

export default ParaphraseForm;
