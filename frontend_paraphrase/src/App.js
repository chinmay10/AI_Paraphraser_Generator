import React from 'react';
class Addition extends React.Component{
    constructor(){
        super();
        this.state={
        text1:'',
        
        total:''
        }
    }

    handlenum1 = (event) => {
        this.setState({
            text1:event.target.value
        })

    }


    exe = (event) => {
        this.setState({total:parseInt(this.state.text1) + 
parseInt(this.state.num2)});
        event.prevent.default();

    }

    render(){
        return(
            <div>
            <h1> Paraphrase Generator </h1>
            <form onSubmit={this.exe}>
            <div>
            Number 01:
            <input type="text" value={this.state.text1} onChange={this.handlenum1}/>
            </div>

            <div>
            <button type= "submit"> Add </button>
            </div>
            </form>
            {this.state.total}
            </div>

        )
    }

}
export default Addition;