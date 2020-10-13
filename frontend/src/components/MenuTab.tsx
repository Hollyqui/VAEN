import React from 'react'
import { Link } from 'react-router-dom'

function MenuTab(props: { name: string, route: string }) {
    return(
        <Link to={props.route}> {props.name} </ Link>
    )
}

export default MenuTab