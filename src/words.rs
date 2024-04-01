use std::io::BufRead as _;

#[derive(Default, Debug)]
pub struct Node {
    pub children: std::collections::HashMap<char, Node>,
    pub is_end: bool,
}

pub fn load() -> (Node, usize) {
    let mut max_len = 0;
    let mut root = Node::default();

    for word in std::io::BufReader::new(std::io::Cursor::new(include_str!("dictionary"))).lines() {
        let word = word.unwrap();
        let mut node = &mut root;
        for letter in word.chars() {
            node = node
                .children
                .entry(letter)
                .or_insert_with(|| Node::default());
        }
        if word.len() > max_len {
            max_len = word.len();
        }
        node.is_end = true;
    }

    (root, max_len)
}
