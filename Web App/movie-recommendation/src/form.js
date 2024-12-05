import {useState} from 'react';
import './form.css'

function Form() {
    const [form, setForm] = useState({
        user_id: "",
        num_recs: ""
    });

    const [result, setResult] = useState("");
    const [loading, setLoading] = useState("");
    const handleSubmit = (event) => {
        event.preventDefault();

        const form_data = new FormData();
        form_data.append("1", form.user_id);
        form_data.append("2", form.num_recs);

        setLoading(true);
        fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            body: form_data
        })
        .then(response => response.text())
        .then(html => {
            setResult(html);
            setLoading(false);
        });
    }
    const handleClear = (event) => {
        setForm({
            user_id: "",
            num_recs: ""
        });
        setResult("");
    }
    const onChange = (event) => {
        const name = event.target.name;
        const value = event.target.value;
        setForm({...form, [name]: value});
    }
    return (
        <form onSubmit={handleSubmit}>
            <input type="number" name="user_id" value={form.user_id} onChange={onChange} placeholder="ex. 23" required></input>
            <input type="number" name="num_recs" value={form.num_recs} onChange={onChange} placeholder="ex. 5" required></input>
            <button type='submit' disabled={loading}>{loading ? "Gathering movies..." : "Submit Form"}</button>
            {result && <span onClick={handleClear}>Clear Form</span>}
            {result && <div dangerouslySetInnerHTML={{__html: result}} className='result'></div>}
        </form>

    );
}

export default Form;