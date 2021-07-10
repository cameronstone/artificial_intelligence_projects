import os
import random
import re
import sys
import copy

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    # dictionary to be returned
    new_dict = {}
    # case where a page is chosen at random from all pages
    num_pages = len(corpus)
    prob_random = (1 - damping_factor) / num_pages
    # add uniform probability to all keys
    for key in corpus.keys():
        new_dict[key] = prob_random
    num_linked = len(corpus[page])

    # if the chosen page has no links, choose from all at random
    if num_linked == 0:
        prob_linked = damping_factor / num_pages
        # loop through keys, assign values
        for key in corpus.keys():
            new_dict[key] = prob_linked
    # if the chosen page has linked pages, choose randomly from those
    else:
        prob_linked = damping_factor / num_linked
        # loop through keys, assign values
        for linked_page in corpus[page]:
            new_dict[linked_page] = new_dict[linked_page] + prob_linked

    return new_dict


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # create dictionary to be returned
    pageranks = {}
    for page in corpus.keys():
        pageranks[page] = 0
    # choose starting page at random for first iteration
    starting_page = random.choice(list(corpus.keys()))
    pageranks[starting_page] = pageranks[starting_page] + 1
    # temp_dict holds dictionary that helps determine next move
    temp_dict = transition_model(corpus, starting_page, damping_factor)

    # iterate through the rest of the number of desired samples
    for _ in range(n - 1):
        # based off probabilities, choose next page to move to
        keys = list(temp_dict.keys())
        weights = list(temp_dict.values())
        next_page = (random.choices(keys, weights=weights, k=1))[0]
        # add 1 to count of that page
        pageranks[next_page] = pageranks[next_page] + 1
        # reset temp_dict to new transition model
        temp_dict = transition_model(corpus, next_page, damping_factor)

    # convert numbers of counts to proportions
    total = 0
    total = sum(pageranks.values())
    pageranks = {page: (count / total) for (page, count) in pageranks.items()}

    # return counts in form of dictionary
    return pageranks


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # create dictionary to be returned
    pageranks = {}
    # add initial starting values of 1/N
    all_pages = corpus.keys()
    n = len(all_pages)
    starting_val = 1 / n
    # set every key to proper starting value
    for page in all_pages:
        pageranks[page] = starting_val

    # recursive iterative algorithm
    def iterative_algorithm(current_page, d):
        first_term = (1 - d) / n
        second_term = 0
        # determine which pages link to desired page, gets PR & NumLinks
        parent_pages = []
        for page in all_pages:
            # set of all outgoing links on this specific page
            outgoing_links = corpus[page]
            # if a page has no links at all, divide evenly across all
            if len(outgoing_links) == 0:
                # n is total number of pages
                parent_pages.append((page, n))
            elif current_page in outgoing_links:
                parent_pages.append((page, len(outgoing_links)))
        # parent_pages has all pages that link to it and their totals
        # summation of term involving pages that link to current_page
        for (parent, total) in parent_pages:
            second_term = second_term + d * pageranks[parent] / total
        return first_term + second_term

    # create dictionary to compare with new
    old_pageranks = {}
    for page in all_pages:
        old_pageranks[page] = 0

    def check_completion(current_dict, old_dict):
        for key in current_dict.keys():
            if abs(current_dict[key] - old_dict[key]) > 0.001:
                return False
        return True

    while check_completion(pageranks, old_pageranks) is False:
        old_pageranks = copy.deepcopy(pageranks)
        for page in all_pages:
            pageranks[page] = iterative_algorithm(page, damping_factor)

    return pageranks


if __name__ == "__main__":
    main()
