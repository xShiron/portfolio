const bcrypt = require("bcrypt");

const pool = require('./connect_db');

async function check_hash() {
    const query = {
        name: 'check-hash',
        text: "SELECT u.password FROM utilizador u WHERE u.nome = 'aaaaa'"
    }
    const res = await pool.query(query);

    bcrypt.compare("TeeTste1234!!!..", res.rows[0].password, (error, result) => {
        console.log(result);
    });
}


check_hash();