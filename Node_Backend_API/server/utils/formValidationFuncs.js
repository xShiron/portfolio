"use strict";

var validator = require('validator');

/**
 * Esta função retira caracteres especiais de tags HTML.
 * @param {string} input 
 * @returns Input limpo.
 */
function limparInput(input){
    return validator.unescape(input);
}

/**
 * Esta função verifica se o nome introduzido utiliza apenas alfanuméricos.
 * @param {string} nomeLimpo 
 * @returns True -> é um nome válido // False -> é um nome inválido.
 */
function validarNome(nomeLimpo){
    return validator.isAlpha(nomeLimpo);
}

/**
 * Esta função verifica se o mail introduzido é um mail.
 * @param {string} mailLimpo
 * @returns True -> é um mail válido // False -> é um mail inválido.
 */
function validarMail(mailLimpo){
    return validator.isEmail(mailLimpo);
}

/**
 * Esta função verifica se a password é forte.
 * @param {string} passLimpo 
 * @returns True -> password forte // False -> password fraca.
 */
function validarPassword(passLimpa){
    return validator.isStrongPassword(passLimpa);
}


module.exports = { validarNome, validarMail, validarPassword, limparInput };