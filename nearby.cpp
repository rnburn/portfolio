#include <iostream>
#include <vector>
#include <cassert>
#include <memory>
#include <cmath>
#include <set>
#include <algorithm>
#include <limits>

typedef uint32_t id_type;
typedef id_type index_type;
typedef std::vector<index_type> IndexMap;
typedef std::vector<id_type> OrderedQuestionList;
typedef std::vector<OrderedQuestionList> TopicQuestionMap;
const size_t MAX_ID = 1e5;
const size_t MAX_QUESTIONS = 1e3;
const double EQUALITY_THRESHOLD = .001;
const size_t MAX_RESULTS_PER_QUERY=100;

struct Topic {
  id_type id;
  double x,y;
};

struct QuoraData {
  QuoraData(id_type ntopics, id_type nquestions, id_type nqueries_) : 
      topics(ntopics),
      topic_index_map(MAX_ID+1),
      question_index_map(MAX_ID+1),
      topic_question_map(ntopics),
      nqueries(nqueries_),
      num_question_linked_topics(0) {}
      
  std::vector<Topic> topics;
  IndexMap topic_index_map;
  IndexMap question_index_map;
  TopicQuestionMap topic_question_map;
  size_t nqueries, num_question_linked_topics;
};

inline
bool return_first(double d1sq, id_type id1, double d2sq, id_type id2) {
  return d1sq < d2sq || (!(d2sq<d1sq) && id1>id2);  
}

typedef std::pair<double, id_type> Resultant;

struct ResultantComparison {
  bool operator()(const Resultant& lhs,
                  const Resultant& rhs) const
  {
    return return_first(rhs.first, rhs.second, lhs.first, lhs.second); 
  }
};

struct ResultantIdComparison {
  bool operator()(const Resultant& lhs,
                  const Resultant& rhs) const
  {
    return lhs.second < rhs.second;
  }
};

typedef std::multiset<Resultant, ResultantComparison> OrderedResultList;

/**
 * Returns the lowest ranked entity in <results>
 */
inline
OrderedResultList::iterator
find_worst(OrderedResultList& results, double greatest_distance) {
  double bound = std::max(greatest_distance - EQUALITY_THRESHOLD, 0.0);
  bound *= bound;
  auto i = results.begin();
  auto last = results.end();
  id_type worst_id = i->second;
  auto worst_iter = i;
  ++i;
  for(; i!=last; ++i) {
      if(i->first >= bound) {
          id_type id = i->second;
          if(id < worst_id) {
              worst_id = id;
              worst_iter = i;
          }
      } else {
          break;
      }
  }
  
  return worst_iter;
}

inline
double compute_distancesq(double x1, double y1, double x2, double y2) {
  double deltax = x2 - x1;
  double deltay = y2 - y1;
  return deltax*deltax+deltay*deltay;
}

/******************************************************************************
 * Reading Input
 *****************************************************************************/

double read_position() {
  double res;
  std::cin >> res;
  return res;
}

void read_counts(size_t& ntopics, size_t& nquestions, size_t& nqueries) {
  std::cin >> ntopics >> nquestions >> nqueries;
  assert(nquestions <= MAX_QUESTIONS);
}

void read_topics(std::vector<Topic>& topics,
                 IndexMap& index_map) 
{
  for(id_type i=0; i<topics.size(); ++i) {
    Topic& topic = topics[i];
    std::cin >> topic.id;
    assert(topic.id <= MAX_ID);
    topic.x = read_position();
    topic.y = read_position();
    index_map[topic.id] = i;
  }
}

void read_questions(size_t nquestions,
                    const IndexMap& topic_index_map,
                    IndexMap& question_index_map,
                    TopicQuestionMap& topic_question_map,
                    size_t& num_question_linked_topics)
{
  for(index_type i=0; i<nquestions; ++i) {
    id_type qid;
    size_t ntopics;
    std::cin >> qid;
    std::cin >> ntopics;
    assert(ntopics <= 10);
    assert(qid <= MAX_ID);
    question_index_map[qid] = i;
    for(size_t j=0; j<ntopics; ++j) {
      id_type tid;
      std::cin >> tid;
      TopicQuestionMap::value_type& qset = topic_question_map[topic_index_map[tid]];
      if(qset.empty())
        ++num_question_linked_topics;
      qset.push_back(qid);
    }
  }
  
  // Since the questions will be iterated through repeatedly, it's best to use a sorted 
  // vector for storage since it's elements can be traversed much faster than a set
  for(auto& qset : topic_question_map)
    std::sort(qset.begin(), qset.end(), std::greater<id_type>());
}

QuoraData read_quora_data() {
  size_t ntopics, nquestions, nqueries;
  read_counts(ntopics, nquestions, nqueries);
  QuoraData qdata(ntopics, nquestions, nqueries);
  read_topics(qdata.topics, qdata.topic_index_map);
  read_questions(nquestions, 
                 qdata.topic_index_map, 
                 qdata.question_index_map, 
                 qdata.topic_question_map,
                 qdata.num_question_linked_topics);
  return qdata;
}

/******************************************************************************
 * K-D Tree Data Structure / Algorithms
 *****************************************************************************/

class KDTree {
  /**
   * The k-d tree makes use of templated doubly recursive functions that are 
   * specific to each axis orientation. This allows us to avoid storing and
   * conditioning on the axis orientation for each node.
   */
public:
  template<class RandomAccessIterator>
  KDTree(RandomAccessIterator first, 
         RandomAccessIterator last);
  
  template<class Visitor>
  void branch_and_bound(Visitor& v, double x, double y) const;
  
private:
  KDTree(const KDTree&); //no copy
  KDTree& operator=(const KDTree&); //no assignment
  
  static const int X_AXIS=0;
  enum class_t {
    LEAF=0,
    LEFT_CHILD,
    RIGHT_CHILD,
    BOTH_CHILDREN
  };
  
  struct Node {
    Topic point;
    class_t type;
    std::unique_ptr<Node> left, right;
  };
  
  std::unique_ptr<Node> root;
  
  static constexpr int other_axis(int axis) { return (axis+1) % 2; } 
  
  template<int>
  struct Comparison {
    bool operator()(const Topic& lhs, const Topic& rhs);
  };
  
  template<class Visitor>
  void examine(Visitor& v, const Topic& t, double x, double y) const;
  
  template<class RandomAccessIterator, int Axis>
  std::unique_ptr<Node> create_kd_tree(RandomAccessIterator first, 
                                       RandomAccessIterator last);
  
  template<class Visitor, int Axis>
  void branch_and_bound(const Node* node, Visitor& v, double x, double y) const;
  
  template<class Visitor, int Axis>
  void branch_and_bound_both(const Node* node, Visitor& v, double x, double y) const;
};

/*** Comparison */
template<>
struct KDTree::Comparison<0> {
  bool operator()(const Topic& lhs, const Topic& rhs) {
    return lhs.x < rhs.x;
  }
};

template<>
struct KDTree::Comparison<1> {
  bool operator()(const Topic& lhs, const Topic& rhs) {
    return lhs.y < rhs.y;
  }
};

/*** compute_delta */
template<int Axis>
inline
double compute_delta(double x1, double y1, double x2, double y2);

template<>
inline
double compute_delta<0>(double x1, double y1, double x2, double y2)
{
  return x2 - x1;
}

template<>
inline
double compute_delta<1>(double x1, double y1, double x2, double y2)
{
  return y2 - y1;
}

/*** create_kd_tree */
template<class RandomAccessIterator>
KDTree::KDTree(RandomAccessIterator first, RandomAccessIterator last) {
  root = create_kd_tree<RandomAccessIterator, X_AXIS>(first, last);
}

template<class RandomAccessIterator, int Axis>
std::unique_ptr<KDTree::Node>
KDTree::create_kd_tree(RandomAccessIterator first,
                       RandomAccessIterator last)
{
  if(first == last)
      return std::unique_ptr<Node>();

    std::unique_ptr<Node> tree(new Node());

    size_t n = last - first;
    if(n == 1) {
      tree->point = *first;
      tree->type=LEAF;
      return tree;
    }

    RandomAccessIterator median = first+n / 2;
    
    assert(first < median && median < last);
    nth_element(first, median, last, Comparison<Axis>());

    tree->point = *median;

    tree->left = create_kd_tree<RandomAccessIterator, other_axis(Axis)>(first, median);

    tree->right = create_kd_tree<RandomAccessIterator, other_axis(Axis)>(median+1, last);
    if(!tree->left.get())
      tree->type = RIGHT_CHILD;
    else if(!tree->right.get())
      tree->type = LEFT_CHILD;
    else
      tree->type = BOTH_CHILDREN;   

    return tree; 
}

/*** branch_and_bound */
template<class Visitor, int Axis>
inline
void KDTree::branch_and_bound_both(const Node* node, Visitor& v, double x, double y) const
{
  double delta1 = compute_delta<Axis>(node->point.x, node->point.y, x, y);
  double delta1sq = delta1*delta1;
  if(delta1 > 0) {
      branch_and_bound<Visitor, other_axis(Axis)>(node->right.get(), v, x, y);
      if(v.bound_check(delta1sq)) {
          double delta2 = compute_delta<other_axis(Axis)>(node->point.x, node->point.y, x, y);
          double dsq = delta1sq+delta2*delta2;
          v.examine(dsq, node->point.id);
          branch_and_bound<Visitor, other_axis(Axis)>(node->left.get(), v, x, y);
      }
  } else {
      branch_and_bound<Visitor, other_axis(Axis)>(node->left.get(), v, x, y);
      if(v.bound_check(delta1sq)) {
          double delta2 = compute_delta<other_axis(Axis)>(node->point.x, node->point.y, x, y);
          double dsq = delta1sq+delta2*delta2;
          v.examine(dsq, node->point.id);
          branch_and_bound<Visitor, other_axis(Axis)>(node->right.get(), v, x, y);             
      }
  }
}

template<class Visitor>
inline
void KDTree::examine(Visitor& v, const Topic& t, double x, double y) const {
  double dsq = compute_distancesq(t.x, t.y, x, y);
  v.examine(dsq, t.id);
}

template<class Visitor, int Axis>
void KDTree::branch_and_bound(const Node* node, Visitor& v, double x, double y) const
{
  switch(node->type) {
  case LEAF:
    examine(v, node->point, x, y);
    break;
  case LEFT_CHILD:
    examine(v, node->left->point, x, y);
    examine(v, node->point, x, y);
    break;
  case RIGHT_CHILD:
    examine(v, node->right->point, x, y);
    examine(v, node->point, x, y);
    break;
  case BOTH_CHILDREN:
    branch_and_bound_both<Visitor, Axis>(node, v, x, y);
    break;
  }
}

template<class Visitor>
inline
void
KDTree::branch_and_bound(Visitor& v, double x, double y) const
{
  if(root.get())
    branch_and_bound<Visitor, X_AXIS>(root.get(), v, x, y);
}

/******************************************************************************
 * Outputing Results
 *****************************************************************************/

inline
bool output_first(double d1, id_type id1, double d2, id_type id2)
{
  return (d1-d2 <= EQUALITY_THRESHOLD) && (id1 > id2);
}

inline 
void insert_resultant(Resultant* xs, size_t i, double dsq, id_type id) 
{
  double d = std::sqrt(dsq);
  xs[i].first = d;
  xs[i].second = id; 
  while(i > 0 && output_first(d, id, xs[i-1].first, xs[i-1].second)) {
    std::swap(xs[i-1], xs[i]);
    --i;
  }

}

/**
 * If the location data is uniformly random, then it's unlikely that
 * the output order will differ from the ordering used in the visitor
 * classes, so I can safely use insertion which runs efficiently for
 * already sorted data.
 * 
 * I assume that the ranking order is transitive for all input given
 * (this isn't true in general) so that each query will have a unique,
 * correct solution.
 */
template<class RandomAccessIterator>
void output_results(RandomAccessIterator first,
                    RandomAccessIterator last)
{
  if(first == last) {
      std::cout << std::endl;
      return;
  }
  Resultant xs[MAX_RESULTS_PER_QUERY];

  size_t nresults=0;
  for(; first!=last; ++first) {
    insert_resultant(xs, nresults++, first->first, first->second);
  }

  std::cout << xs[0].second;
  for(size_t i=1; i<nresults; ++i)
    std::cout << " " << xs[i].second;
  std::cout << std::endl; 
}

/******************************************************************************
 * Topic Visitor
 *****************************************************************************/

class TopicVisitor {
public:
  TopicVisitor(size_t nresults_) : 
    nresults(0),
    max_results(nresults_),
    distancesq_bound(std::numeric_limits<double>::max()),
    greatest_distance(std::numeric_limits<double>::max()),
    greatest_distancesq(std::numeric_limits<double>::max()) {}
  bool bound_check(double bound) const;
  void examine(const Topic& t);
  void examine(double dsq, id_type id);
  void write_output();
private:
  void set_bound();
  void pop_worst();
  OrderedResultList results;
  size_t nresults;
  size_t max_results;
  double distancesq_bound, 
         greatest_distance, 
         greatest_distancesq;
};

inline
void TopicVisitor::write_output() {
  output_results(results.rbegin(), results.rend());
}

inline
bool TopicVisitor::bound_check(double bound) const
{
  return bound <= distancesq_bound;
}

inline
void TopicVisitor::set_bound() {
  double new_greatest_distancesq = results.begin()->first;
  if(new_greatest_distancesq == greatest_distancesq)
    return;
  else
    greatest_distancesq = new_greatest_distancesq;
  greatest_distance = std::sqrt(greatest_distancesq);
  double x = greatest_distance + EQUALITY_THRESHOLD;
  distancesq_bound = x*x;
  
}

inline
void TopicVisitor::pop_worst() {
  OrderedResultList::iterator i = find_worst(results, greatest_distance);
  results.erase(i);
  set_bound();
}

inline
void TopicVisitor::examine(double dsq, id_type id)
{  
  if(nresults < max_results) {
      Resultant elem(dsq, id);
      results.insert(elem);
      ++nresults;
      if(nresults == max_results)
        set_bound();
  } else if(dsq <= distancesq_bound) {
      Resultant elem(dsq, id);
      results.insert(elem);
      pop_worst();
     
  }
}

/******************************************************************************
 * Question Visitor
 *****************************************************************************/

class QuestionVisitor {
public:
  QuestionVisitor(const IndexMap& tindex_,
                  const IndexMap& qindex_, 
                  const TopicQuestionMap& qlist_, 
                  size_t max_results_) :
      tindex(tindex_),
      qindex(qindex_),
      qlist(qlist_),
      nresults(0),
      max_results(max_results_),
      qiter_map(MAX_QUESTIONS, results.end()),
      distancesq_bound(std::numeric_limits<double>::max()),
      greatest_distance(std::numeric_limits<double>::max()),
      greatest_distancesq(std::numeric_limits<double>::max()) {}
  
  bool bound_check(double dsq) const;
  void examine(double dsq,  id_type id);
  void write_output();
private:  
  const IndexMap& tindex;
  const IndexMap& qindex;
  const TopicQuestionMap& qlist;
  typedef std::vector<OrderedResultList::iterator> QuestionMap;
  OrderedResultList results;
  QuestionMap qiter_map;
  
  size_t nresults;
  size_t max_results;
  double distancesq_bound, 
         greatest_distance, 
         greatest_distancesq;
  
  void insert_element(double dsq, id_type id);
  bool insert_remove_element(double dsq, id_type id);
  void insert_range(double dsq,
                    OrderedQuestionList::const_iterator first,
                    OrderedQuestionList::const_iterator last);
  void insert_remove_range(double dsq,
                           OrderedQuestionList::const_iterator first,
                           OrderedQuestionList::const_iterator last);  
  void set_bound();
  id_type pop_worst();
};

inline
bool QuestionVisitor::bound_check(double bound) const
{
  return bound <= distancesq_bound;
}

inline
void QuestionVisitor::set_bound() {
  double new_greatest_distancesq = results.begin()->first;
  if(new_greatest_distancesq == greatest_distancesq)
    return;
  else
    greatest_distancesq = new_greatest_distancesq;
  greatest_distance = std::sqrt(greatest_distancesq);
  double x = greatest_distance + EQUALITY_THRESHOLD;
  distancesq_bound = x*x;
  
}

inline
id_type QuestionVisitor::pop_worst() {
  OrderedResultList::iterator i = find_worst(results, greatest_distance);
  id_type id = i->second;
  results.erase(i);
  qiter_map[qindex[id]] = results.end();
  return id;
}

inline
void QuestionVisitor::insert_element(double dsq, id_type id) {
  OrderedResultList::iterator& i = qiter_map[qindex[id]];  
  Resultant elem(dsq, id);
  if(i != results.end()) {
      if(i->first < dsq)
        return;
      results.erase(i);
      i = results.insert(elem);
  } else {
      ++nresults;
      i = results.insert(elem);
  }
}

inline
bool QuestionVisitor::insert_remove_element(double dsq, id_type id) {
  OrderedResultList::iterator& i = qiter_map[qindex[id]];  
    
  Resultant elem(dsq, id);
  if(i != results.end()) {
      if(i->first < dsq)
        return true;
      results.erase(i);
      i = results.insert(elem);
      return true;
  } else {
      i = results.insert(elem);
      if(pop_worst() == id)
        return false;
      else
        return true;
  }
}

inline 
void QuestionVisitor::insert_range(double dsq,
                                   OrderedQuestionList::const_iterator first,
                                   OrderedQuestionList::const_iterator last)
{
  for(; first!=last; ++first) {
      insert_element(dsq, *first);
      if(nresults == max_results) {
          ++first;
          insert_remove_range(dsq, first, last);
          return;
      }
  }
}

inline
void QuestionVisitor::insert_remove_range(double dsq,
                                          OrderedQuestionList::const_iterator first,
                                          OrderedQuestionList::const_iterator last)
{
  for(; first!=last; ++first) {
      if(!insert_remove_element(dsq, *first)) {
        return;
      }
  }
}

inline
void QuestionVisitor::examine(double dsq, id_type id) {
  if(nresults < max_results) {
      const TopicQuestionMap::value_type& qset = qlist[tindex[id]];
      insert_range(dsq, qset.begin(), qset.end());
        
      if(nresults == max_results) {
          set_bound();
      }
  } else if(dsq <= distancesq_bound) {
      const TopicQuestionMap::value_type& qset = qlist[tindex[id]];
      insert_remove_range(dsq, qset.begin(), qset.end());
      set_bound();
  }
}

inline
void QuestionVisitor::write_output() {
  output_results(results.rbegin(), results.rend());
}

/******************************************************************************
 * Query Processor
 *****************************************************************************/

void process_topic_query(const KDTree& kdtree, size_t nresults, double x, double y)
{
  TopicVisitor v(nresults);

  kdtree.branch_and_bound(v, x, y);
  v.write_output();
}

void process_question_query(const QuoraData& qdata, const KDTree& kdtree, size_t nresults, double x, double y) {
  QuestionVisitor v(qdata.topic_index_map,
                    qdata.question_index_map,
                    qdata.topic_question_map,
                    nresults);
  kdtree.branch_and_bound(v, x, y);
  v.write_output();
}

inline
void process_query(const QuoraData& qdata, 
                   const KDTree& topic_tree, 
                   const KDTree& question_tree) 
{
  char c;
  size_t nresults;
  double x,y;
  std::cin >> c >> nresults;
  x = read_position();
  y = read_position();
  assert(nresults <= 100);
  if(nresults == 0) {
    std::cout << std::endl;
    return;
  }

  if(c == 't')
    process_topic_query(topic_tree, nresults, x, y);
  if(c == 'q')
    process_question_query(qdata, question_tree, nresults, x, y);
}

inline
void process_queries(const QuoraData& qdata, 
                     const KDTree& topic_tree, 
                     const KDTree& question_tree,
                     size_t nqueries)
{
  for(size_t i=0; i<qdata.nqueries; ++i)
    process_query(qdata, topic_tree, question_tree);  
}

/******************************************************************************
 * Main
 *****************************************************************************/

inline
std::vector<Topic> filter_topics(const std::vector<Topic>& topics, 
                                 const IndexMap& topic_index_map,
                                 const TopicQuestionMap& topic_question_map,
                                 size_t num_question_linked_topics)
{
  std::vector<Topic> res;
  res.reserve(num_question_linked_topics);
  auto filter = [&](const Topic& t) {
    index_type index = topic_index_map[t.id];
    return !topic_question_map[index].empty();   
  };
  std::copy_if(topics.begin(), topics.end(),
               std::back_inserter(res),
               filter);
  
  return res;
}

int main() {
  QuoraData qdata = read_quora_data();
  KDTree topic_tree(qdata.topics.begin(), qdata.topics.end());
  
  // for question queries, a k-d tree indexed on all topics will
  // perform poorly if most topics are not associated with questions;
  // hence, i create a separate k-d tree of only question linked topics
  // for this case
  if(qdata.num_question_linked_topics == qdata.topics.size()) {
      process_queries(qdata, topic_tree, topic_tree, qdata.nqueries);
  } else {
      std::vector<Topic> filtered_topics = filter_topics(qdata.topics, 
                                                         qdata.topic_index_map,
                                                         qdata.topic_question_map,
                                                         qdata.num_question_linked_topics);
      KDTree question_tree(filtered_topics.begin(), filtered_topics.end());
      process_queries(qdata, topic_tree, question_tree, qdata.nqueries);
  }
  return 0;
}
