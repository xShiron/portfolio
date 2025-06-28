"use strict";

let queryLoginMail = (mail) => {
    return {
        text: "SELECT * FROM user WHERE email=$1",
        values: [mail]
    };
}

let queryLoginPass = (pass) => {
    return {
        text: "SELECT * FROM user WHERE pass=$1",
        values: [pass]
    }
}

module.exports = { queryLoginMail, queryLoginPass }