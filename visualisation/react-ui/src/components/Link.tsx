import React from 'react';

function Link(props:{id: number, avg_weight:string, abs_weight: string}){
    
    return (
        <div className="link-div" id={`link-${props.id}`}>

        </div>
    )
}

export default Link;