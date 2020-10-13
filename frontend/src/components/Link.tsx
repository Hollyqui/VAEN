import React, { ReactElement } from 'react';

function Link(props:{id: number, avg_weight:string, abs_weight: string}):ReactElement{
    
    return (
        <div className="link-div" id={`link-${props.id}`}>

        </div>
    )
}

export default Link;