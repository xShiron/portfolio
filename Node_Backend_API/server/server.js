const express = require('express');
const app = express();
const session = require('express-session');
const cookieParser = require('cookie-parser')
const bcrypt = require('bcrypt');
const bodyParser = require('body-parser');
const path = require('path');
const cors = require('cors')
const validator = require('validator');

const dbClient = require('./connect_db');

app.use(bodyParser.urlencoded({ extended: true }));
app.use(express.json()) ;
app.use(cookieParser());
app.use(cors())

// genuuid em falta? 
app.use(session({
    secret: "algo",
    resave: true,
    saveUninitialized: true
}));


app.get("/", (req, res) => {
    req.session.secret = req.sessionID;
});

app.get("/getSession", (req, res) => {
    res.json( req.session.utilizador );
});

//Função para encriptar a password do utilizador
async function passHash(password){
    try{
        // [Salting] 
        // -técnica de segurança que adiciona uma string aleatória (salt) 
        //  à senha antes de aplicar o algoritmo de hash. 
        // [Salt rounds] 
        // - número de interações do algoritmo de hashing
        // - quantas mais salt rounds ->  + segurança && - performance
        const saltRounds = 10;
        const hash = await bcrypt.hash(password, saltRounds);
        return hash;
    }catch(error){
        console.error("Erro ao gerar hash: " + error);
    }
}

//Função para validar os dados de registo
function validateData(data){
    
    const validacaoDados = {
        nome: validator.isAlpha(data.nome),
        mail: validator.isEmail(data.mail), 
        telemovel: validator.isMobilePhone(data.telemovel.toString(), "pt-PT"), //Ex: +351911234567
        pass: validator.isStrongPassword(data.pass),
        nif: data.nif.toString().length == 9, //validator.isVat() so aceita entradas do tipo PT123456789
        nic: validator.isNumeric(data.nic.toString()) && data.nic.toString().length === 9,
        gen: data.gen === "m" || data.gen === "f" || data.gen === "o" || data.gen === "pnd",
        //morada: null,
        dnasc: validator.isDate(data.dnasc)
    }

    return validacaoDados;
}
// TODO:
// - Verificar se se deve declarar a func de callback do routing como async, porque bcrypt só funciona como asincrono
// - Verificar qual método dá mais jeito para o switch da query: [Com(devolve >1 tipo de erro) ou Sem(devolve 1 erro apenas)] break;
// - Interface da página.

app.post("/register", async (req, res) => {
    try{

        // ###########################################################
        // #### 1. Validação e limpeza de dados; Hashing da pass. ####
        // ###########################################################

        //Objeto com cópia da receção do pedido
        const parametros = { ...req.body }

        //Validação de dados
        const validacaoDados = validateData(parametros);

        //Se todos os dados forem validados
        if(Object.values(validacaoDados).every(value => value === true)){
            
            //Limpeza de caracteres
            Object.keys(parametros).forEach((key) => {
                if(typeof parametros[key] === "string"){
                    parametros[key] = validator.escape(validator.trim(parametros[key]));
                } 
            });

            //Hash da password
            parametros.pass = await passHash(parametros.pass);

            //Objeto com informação para o front-end
            const statusMessage = {};
            
            console.log(parametros)

        // #######################################
        // #### 2. Inserção na base de dados. ####
        // #######################################
            const {nif, nic, nome, gen, dnasc, telemovel, mail, pass} = parametros
            //Query registo
            const query = {
                name: 'register-user',
                text: 'INSERT INTO utilizador(nif, nic, nome, genero, dnasc, telemovel, email, password) \
                                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)',
                values: [nif,nic,nome,gen,dnasc,telemovel,mail,pass]
            } 
            dbClient.query(query, (error, results) => {
                        //Caso aconteça um erro na inserção dos dados na BD
                        if(error){
                            console.error(error);
                            switch (error.code){
                                case '23505': //<UNIQUE> error
                                    res.status(409);
                                    const constraintTypes = {
                                        utilizador_nif_pkey: "userDuplicado",
                                        unique_mail_constraint: "mailDuplicado",
                                        unique_nic_constraint: "nicDuplicado",
                                        unique_phone_constraint: "telemovelDuplicado"
                                    }
                                    
                                    const constraintType = constraintTypes[error.constraint];
                                    statusMessage[constraintType] = true;
                                    break;

                                case '23502': //<NOT NULL> error
                                    res.status(422);

                                    const notNullErrors = {
                                        nic: 'notNullnic',
                                        nif: 'notNullnif',
                                        email: 'notNullmail',
                                        nome: 'notNullnome',
                                        dnasc: 'notNulldnasc',
                                        telemovel: 'notNulltelemovel',
                                        password: 'notNullpass',
                                        genero: 'notNullgen'
                                    };

                                    const errorType = notNullErrors[error.column];
                                    statusMessage[errorType] = true;
                                    break;

                                default: //Outros tipos de erros
                                    res.status(500);
                                    statusMessage.erroInterno = true;
                                    break;
                            }

                            //Devolver a respetiva mensagem de erro para ser tratada no front-end
                            res.send(statusMessage);

                            } else { //Utilizador inserido com sucesso
                                res.status(201);
                                statusMessage.userCriado = true;
                                res.send(statusMessage);
                            }                    
                        });
        } else { //Se a validação dos dados falhar
            res.status(400);
            console.log(validacaoDados);
            const errosDetetados = Object.keys(validacaoDados).filter(key => !validacaoDados[key]);
            res.send(errosDetetados);
        }

    } catch (error) { //Se o routing falhar
        console.error(error);
        res.status(404).send("Recurso não encontrado");    
    }

});

app.post("/login", (req, res) => {
    try {
        const {mail, pass} = req.body;
        console.log( req.body );
        console.log( "Info de login recebida: Email - " + mail + " Pass - " + pass);

        if ( true ) {
            res.status(200).send();
            // Criar session aqui.
        } else {
            res.status(401).send(); 
        }


    } catch (error) {
        res.status(500).send();
        console.log("Erro no /login " + error);
    }
});

app.listen(3000, (err) => {
    if ( err ) console.log(err);
    console.log("Servidor a correr.");
});