import React, { useState } from 'react';
import axios from 'axios';
import { Form, Input, Button } from 'antd';

function App() {
  const [form] = Form.useForm();
  const [paraphrasedTexts, setParaphrasedTexts] = useState([]);

  const handleSubmit = async () => {
    try {
      const values = await form.validateFields();
      const response = await axios.post('/api/paraphrase', {
        text: values.text,
        num_paraphrases: 5
      });
      setParaphrasedTexts(response.data);
    } catch (error) {
      console.error(error);
    }
  };

  return (
    <Form form={form} onFinish={handleSubmit}>
      <Form.Item label="Text to paraphrase" name="text">
        <Input />
      </Form.Item>
      <Form.Item>
        <Button type="primary" htmlType="submit">
          Paraphrase
        </Button>
      </Form.Item>
      {paraphrasedTexts.map(text => (
        <div key={text.text}>
          <p>{text.text}</p>
          <p>Similarity score: {text.similarity_score}</p>
        </div>
      ))}
    </Form>
  );
}

export default App;
