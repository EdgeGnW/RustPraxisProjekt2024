use std::fs;
use qwt::QWT256;

fn main() {
    let input : String = fs::read_to_string("input/input.txt").unwrap_or_default();
    parse_input(input);
}

fn parse_input(input : String) -> Vec<String> {
    let mut chars = input.chars().peekable();

    while chars.peek() != None {
        if let Some(c) = chars.next() {
            match c {
                '(' => {
                    let nodes = chars.clone().take_while(|x| *x != ')').collect::<String>();
                    println!("{}", nodes);
                }
                _ => {},
            }
        }
    }
    todo!()
}
