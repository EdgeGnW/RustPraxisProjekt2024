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

#[derive(Debug)]
struct Node {
    identifier : String,
    edges : Vec<String>,
}

fn main() {
    let input : String = fs::read_to_string("input/input.txt").unwrap_or_default();
    let tokens = tokenize_input(input);
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
    fn parse_node(tokens : Vec<Token>) -> Node {
        let mut token_iter = tokens.into_iter().peekable();
        let mut node = Node {identifier: "".to_string(), edges:  Vec::new()};

        while let Some(token) = token_iter.next() {
            match token {
                Token::OpenParen(_) => {
                    let mut ident = String::new();
                    if let Some(Token::Identifier(name)) = token_iter.next() {
                        ident = name;
                    }
                    let mut edges_vec = Vec::new();

                    while let Some(token) = token_iter.next() {
                        match token {
                            Token::Identifier(ident) => {
                                edges_vec.push(ident);
                            },
                            _ => {},
                        }
                    }

                    node.identifier = ident;
                    node.edges = edges_vec;

               },
                _ => {},
            }
        }

        return node;
    }

    fn parse_vector(tokens : Vec<Token>) -> Vec<Node> {
        let mut token_iter = tokens.into_iter();
        let mut node_tokens = Vec::new();
        let mut nodes = Vec::new();

        while let Some(token) = token_iter.next() {
            match token {
                Token::OpenParen(_) => {
                    node_tokens = Vec::new();
                    node_tokens.push(token);
                },
                Token::CloseParen(_) => {
                    node_tokens.push(token);
                    nodes.push(parse_node(node_tokens.clone()));
                    node_tokens = Vec::new();
                },
                _ => {
                    node_tokens.push(token);
                },
            }
        }

        nodes
    }

    let mut ident_map : HashMap<String, i8> = HashMap::new();
    let mut ident_idx = 1;
    
    for token in tokens.clone() {
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
    
    let node_vec = parse_vector(tokens[1..tokens.len()-1].to_vec());
    println!("{:?}", node_vec);
}
