//! Sparse Graphical Model (SGM) implementation
//!
//! This module contains the implementation of Sparse Graphical Model (SGM) as
//! described in the paper "Sparse Graphical Memory for Robust Planning" by
//! Laskin et al. (2020).
//!
//! The SGM is implemented as a component that can be used in any off-policy
//! algorithm, by simply providing a method on top of a [`ReplayBuffer`].
//!
//! The SGM is used in the [`crate::agents::DDPG_SGM`] algorithm.
use {
    crate::{
        envs::TensorConvertible,
        components::ReplayBuffer,
    },
    ordered_float::OrderedFloat,
    petgraph::{
        dot::Dot,
        stable_graph::{
            NodeIndex,
            StableGraph,
        },
        Directed,
    },
    serde::{
        Serialize,
        Deserialize,
    },
    std::{
        collections::HashMap,
        fmt::{
            Debug,
            Display,
        },
        hash::Hash,
    },
};


#[derive(Clone, Copy, Serialize, Deserialize)]
pub enum DistanceMode {
    True,
    Estimated,
}

impl Display for DistanceMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DistanceMode::True => write!(f, "True"),
            DistanceMode::Estimated => write!(f, "Estimated"),
        }
    }
}

/// Return a dotviz representation of the given graph.
pub fn dot<S: Debug>(graph: &StableGraph<S, OrderedFloat<f64>, Directed>) -> String {
    format!("{:?}", Dot::new(graph)).to_string()
}

/// Get edges between candidate node and other nodes.
#[allow(clippy::type_complexity)]
pub fn get_edges<S, D>(
    graph: &StableGraph<S, OrderedFloat<f64>, Directed>,
    indices: &HashMap<S, NodeIndex>,
    s1: &S,
    d: D,
    maxdist: f64,
) -> (Vec<(NodeIndex, OrderedFloat<f64>)>, Vec<(NodeIndex, OrderedFloat<f64>)>)
where
    S: Clone + Eq + Hash,
    D: Fn(&S, &S) -> f64,
{
    let mut edges_from: Vec<(NodeIndex, OrderedFloat<f64>)> = Vec::new();
    let mut edges_to: Vec<(NodeIndex, OrderedFloat<f64>)> = Vec::new();

    for s2 in graph.node_weights() {
        // no self edges
        if s1 == s2 {
            continue;
        }

        let d_out = d(s1, s2);
        let d_in = d(s2, s1);

        if d_out < maxdist && d_in < maxdist {
            edges_to.push((indices[s2], OrderedFloat(d_out)));
            edges_from.push((indices[s2], OrderedFloat(d_in)));
        }
    }
    (edges_from, edges_to)
}

/// Returns the edges to add to the graph if s1 is TWC-consistent.
#[allow(clippy::type_complexity)]
pub fn is_two_consistent<S, D>(
    graph: &StableGraph<S, OrderedFloat<f64>, Directed>,
    s1: &S,
    d: D,
    tau: f64,
) -> bool
where
    S: Clone + Eq + Hash,
    D: Fn(&S, &S) -> f64,
{
    // check if new node is TWC consistent
    graph
        .node_weights()
        .all(|s2| {
            let c_out = *graph.node_weights().map(|w|
                OrderedFloat((d(s1, w) - d(s2, w)).abs())
            ).max().unwrap();
            let c_in = *graph.node_weights().map(|w|
                OrderedFloat((d(w, s1) - d(w, s2)).abs())
            ).max().unwrap();

            c_out >= tau && c_in >= tau
        })
}

/// Add the node and its edges to the graph.
pub fn add_node_to_graph<S>(
    graph: &mut StableGraph<S, OrderedFloat<f64>, Directed>,
    indices: &mut HashMap<S, NodeIndex>,
    s1: &S,
    edges_from: Vec<(NodeIndex, OrderedFloat<f64>)>,
    edges_to: Vec<(NodeIndex, OrderedFloat<f64>)>
)
where
    S: Clone + Eq + Hash
{
    // add node
    let i1 = graph.add_node(s1.clone());
    indices.insert(s1.clone(), i1);

    // add edges
    for (i2, weight) in edges_from {
        graph.add_edge(i2, i1, weight);
    }
    for (i2, weight) in edges_to {
        graph.add_edge(i1, i2, weight);
    }
}

/// Replenish edges of graph.
pub fn edges_to_replenish<S, D>(
    graph: &StableGraph<S, OrderedFloat<f64>, Directed>,
    indices: &HashMap<S, NodeIndex>,
    d: D,
    maxdist: f64,
) -> Vec<(NodeIndex, NodeIndex, OrderedFloat<f64>)>
where
    S: Clone + Eq + Hash,
    D: Fn(&S, &S) -> f64,
{
    let mut edges_list = Vec::new();

    for s1 in graph.node_weights() {
        let (edges_from, edges_to) = get_edges(graph, indices, s1, &d, maxdist);
        edges_list.push((indices[s1], edges_from, edges_to));
    }

    let mut edges_to_replenish = Vec::new();

    for (i1, edges_from, edges_to) in edges_list {

        for (i2, weight) in edges_from {
            if !graph.contains_edge(i2, i1) {
                edges_to_replenish.push((i2, i1, weight));
            }
        }
        for (i2, weight) in edges_to {
            if !graph.contains_edge(i1, i2) {
                edges_to_replenish.push((i1, i2, weight));
            }
        }
    }
    edges_to_replenish
}

impl ReplayBuffer {
    /// Construct a sparse graph from the replay buffer.
    ///
    /// The resulting graph can be seen as a tau-approximate, Q-irrelevant
    /// abstraction. This means that as we vary tau, we vary the sparsity of
    /// the graph in a way that is irrelevant for the Q-function, even in the
    /// goal-conditioned setting.
    ///
    /// # Arguments
    ///
    /// * `d` - The distance function.
    /// * `maxdist` - The maximum distance between two nodes in the graph.
    /// * `tau` - The tau parameter to vary the graph sparsity.
    pub fn construct_sgm<S, D>(
        &self,
        d: D,
        maxdist: f64,
        tau: f64,
    ) -> (
        StableGraph<S, OrderedFloat<f64>, Directed>,
        HashMap<S, NodeIndex>,
    )
    where
        S: Clone + Eq + Hash + TensorConvertible,
        D: Fn(&S, &S) -> f64,
    {
        // initialize the SGM data structures
        let mut graph: StableGraph<S, OrderedFloat<f64>, Directed> = StableGraph::default();
        let mut indices: HashMap<S, NodeIndex> = HashMap::new();

        // iterate over the set of nodes in the buffer
        for s1 in self.all_states::<S>().iter() {

            if is_two_consistent(&graph, s1, &d, tau) {
                let (edges_from, edges_to) = get_edges(&graph, &indices, s1, &d, maxdist);
                add_node_to_graph(&mut graph, &mut indices, s1, edges_from, edges_to);
            }
        }

        (graph, indices)
    }
}
