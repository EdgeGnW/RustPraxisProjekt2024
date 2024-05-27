use std::fs;
use std::collections::HashMap;
use qwt::QWT256;

struct Graph_Wavelet_Tree {
    tree : QWT256<i8>,
    bitmap : Vec<i8>,
}

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
enum Token {
    Identifier(String),
    OpenParen(char),
    CloseParen(char),
    OpenSquare(char),
    CloseSquare(char),
    Comma(char),
}

fn main() {
    let input : String = fs::read_to_string("input/input.txt").unwrap_or_default();
    let tokens = tokenize_input(input);
    println!("{:?}", &tokens);
    parse_tokens(tokens);
}

fn tokenize_input(input : String) -> Vec<Token> {
    let mut tokens : Vec<Token> = Vec::<Token>::new();
    
    let mut chars = input.chars().peekable();

    loop {
        if let Some(c) = chars.next() {
            match c {
                '(' => tokens.push(Token::OpenParen(c)),
                ')' => tokens.push(Token::CloseParen(c)),
                '[' => tokens.push(Token::OpenSquare(c)),
                ']' => tokens.push(Token::CloseSquare(c)),
                ',' => tokens.push(Token::Comma(c)),
                'A'..='Z' | 'a'..='z' | '0'..='9' | '_' => {
                    let mut identifier : String = String::new();
                    identifier.push(c);
                    while let Some(ident) = chars.next_if(|x| x.is_alphabetic() || x.is_alphanumeric()) {
                        identifier.push(ident);
                    };

                    tokens.push(Token::Identifier(identifier));
                },
                _ => {
                    while let Some(_) = chars.next_if(|x| x.is_whitespace()) {};
                }
                
            }
        } else {
            break;
        }
    }
    tokens
}

fn parse_tokens(tokens : Vec<Token>) {
    fn parse_tokens_impl(token : Token) {
        todo!()
    }

    let mut ident_map : HashMap<String, i8> = HashMap::new();
    let mut ident_idx = 1;
    
    for token in tokens {
        match token {
            Token::Identifier(name) => {
                if let None = ident_map.get(&name) {
                    ident_map.insert(name.clone(), ident_idx);
                    ident_idx += 1;
                }
            },
            _ => {},
        }
    }

    todo!();
}
