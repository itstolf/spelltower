use std::io::BufRead as _;

#[derive(Default, Debug)]
pub struct Node {
    children: [Option<Box<Node>>; 26],
    is_end: bool,
}

impl Node {
    pub fn get(&self, c: char) -> Option<&Node> {
        self.children
            .get(c as usize - 'A' as usize)
            .map(|v| v.as_ref().map(|v| v.as_ref()))
            .flatten()
    }

    pub fn is_end(&self) -> bool {
        self.is_end
    }
}

pub fn load() -> (Node, usize) {
    let mut max_len = 0;
    let mut root = Node::default();

    for word in std::io::BufReader::new(std::io::Cursor::new(include_str!("dictionary"))).lines() {
        let word = word.unwrap();
        let mut node = &mut root;
        for letter in word.chars() {
            node = node.children[letter as usize - 'A' as usize]
                .get_or_insert_with(|| Box::new(Node::default()));
        }
        if word.len() > max_len {
            max_len = word.len();
        }
        node.is_end = true;
    }

    (root, max_len)
}
