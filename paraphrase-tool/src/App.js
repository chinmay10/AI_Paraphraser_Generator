import { Form, Input, Button } from 'antd';
import axios from 'axios';
import { useState } from 'react';

function ParaphraseForm() {
  const [form] = Form.useForm();
  const [paraphrasedTexts, setParaphrasedTexts] = useState([]);

  const handleSubmit = async () => {
    try {
      const values = await form.validateFields();
      const response = await axios.post('/api/paraphrase', values);
      const paraphrasedTexts = response.data;  // This should be an array of objects with 'text' and 'similarity' properties
      setParaphrasedTexts(paraphrasedTexts);
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
      {paraphrasedTexts.map(({ text, similarity }) => (
        <div key={text}>
          {text} ({similarity}%)
        </div>
      ))}
    </Form>
  );
}
